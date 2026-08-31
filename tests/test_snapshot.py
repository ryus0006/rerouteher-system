"""Snapshot orchestration tests: skill extraction, occupation tiers, reframe, normalization.

Repos and models are faked so this runs without a DB or torch. Tier 1 (TF-IDF/ESCO) is
logged only; Tier 2 (embedding over roles) is the response source.
"""
import numpy as np
import pytest

from app.config import Settings
from app.repositories import caregiving as caregiving_repo
from app.repositories import roles as roles_repo
from app.repositories import skills as skills_repo
from app.repositories.caregiving import ReframedRow
from app.repositories.roles import NearestRole, Role
from app.repositories.skills import SkillMatch, SkillRow
from app.schemas.cv import CV, Experience
from app.schemas.snapshot import Break, SnapshotRequest
from app.services.occupation_matcher import OccupationMatch
from app.services.snapshot import SnapshotService

pytestmark = pytest.mark.asyncio

SKILL_ROWS = [
    SkillRow("s_pm", "Project management", "soft"),
    SkillRow("s_budget", "Budgeting", "soft"),
    SkillRow("s_ux", "User research", "technical"),
]

# (skill_id, term) pairs: canonical names + aliases, as load_alias_dictionary returns
ALIAS_PAIRS = [
    ("s_pm", "Project management"),
    ("s_pm", "project coordination"),
    ("s_budget", "Budgeting"),
    ("s_budget", "budget management"),
    ("s_ux", "User research"),
    ("s_ux", "ux research"),
]


class FakeEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 384), dtype="float32")

    def encode_one(self, text):
        return np.zeros(384, dtype="float32")


class FakeMatcher:
    """Records what Tier 1 was called with; return value is only logged."""

    def __init__(self):
        self.calls = []

    def predict(self, job_title, skills, work_length_years=None, top_k=3):
        self.calls.append({"job_title": job_title, "skills": skills})
        return [OccupationMatch("1234.1", "Some ESCO Role", "1234", 0.9, "tfidf_logreg")]


@pytest.fixture(autouse=True)
def patch_repos(monkeypatch):
    async def _list_skills(session):
        return SKILL_ROWS

    async def _load_alias_dictionary(session):
        return ALIAS_PAIRS

    async def _match_by_embedding(session, vec, k, threshold):
        return [SkillMatch("s_ux", "User research", 0.72)]

    async def _reframe(session, activities):
        rows = []
        for a in activities:
            rows.append(ReframedRow(a, "Coordination", "Coordination"))
            rows.append(ReframedRow(a, "Coordination", "Coordination"))  # duplicate -> dedupe target
        return rows

    async def _nearest(session, vec, k):
        return [
            NearestRole("r1", "Project Coordinator", 0.91),
            NearestRole("r2", "Operations Executive", 0.77),
            NearestRole("r3", "Admin Executive", 0.66),
            NearestRole("r4", "HR Coordinator", 0.60),
        ]

    async def _get_by_esco(session, code, esco_title=None):
        # the FakeMatcher's esco_code resolves to a role
        return Role("r_mkt", "Marketing Manager", "1234") if code == "1234.1" else None

    async def _get_by_masco(session, code):
        return Role("r_mkt", "Marketing Manager", "1234") if code == "1234" else None

    monkeypatch.setattr(skills_repo, "list_skills", _list_skills)
    monkeypatch.setattr(skills_repo, "load_alias_dictionary", _load_alias_dictionary)
    monkeypatch.setattr(skills_repo, "match_by_embedding", _match_by_embedding)
    monkeypatch.setattr(caregiving_repo, "reframe", _reframe)
    monkeypatch.setattr(roles_repo, "nearest_by_embedding", _nearest)
    monkeypatch.setattr(roles_repo, "get_by_esco_code", _get_by_esco)
    monkeypatch.setattr(roles_repo, "get_by_masco_code", _get_by_masco)


def _request(title="Project Coordinator") -> SnapshotRequest:
    cv = CV(
        raw_text="Led project management and budgeting for teams.",
        experiences=[Experience(title=title, description="Led project management.")],
        skill_mentions=["budgeting"],
    )
    return SnapshotRequest(cv=cv, break_=Break(duration_years=3, activities=["Ran the household"]))


async def test_skills_and_embedding_response():
    svc = SnapshotService(Settings(), embedder=FakeEmbedder(), tfidf_matcher=None)
    resp = await svc.generate(_request(), session=object())

    names = {p.skill for p in resp.professional_skills}
    assert "Project management" in names  # exact alias hit in raw_text
    assert "Budgeting" in names           # from skill_mentions
    assert "User research" in names       # semantic pass
    assert resp.previous_occupation.method == "embedding"
    assert resp.previous_occupation.role == "Project Coordinator"
    roles = [r.role for r in resp.recommended_roles]
    assert roles == ["Project Coordinator", "Operations Executive", "Admin Executive"]
    assert resp.recommended_roles[0].similarity == 1.0


async def test_tier1_masco_code_maps_to_role():
    # FakeMatcher returns masco "1234" -> resolves to a role in the role table
    matcher = FakeMatcher()
    svc = SnapshotService(Settings(), embedder=FakeEmbedder(), tfidf_matcher=matcher)
    resp = await svc.generate(_request(), session=object())

    assert matcher.calls, "Tier 1 matcher should be invoked"
    assert resp.previous_occupation.role == "Marketing Manager"  # from MASCO lookup, not embedding
    assert resp.previous_occupation.method == "classifier"
    assert resp.recommended_roles[0].role == "Marketing Manager"
    assert resp.recommended_roles[0].similarity == 1.0
    # only one MASCO match resolved -> topped up to 3 with nearest roles by embedding
    assert len(resp.recommended_roles) == 3
    assert resp.recommended_roles[1].role == "Project Coordinator"


async def test_falls_back_to_embedding_when_masco_unresolved():
    # matcher returns a masco code that does not resolve -> Tier 2 embedding
    class NoMatchMatcher:
        def predict(self, job_title, skills, work_length_years=None, top_k=3):
            return [OccupationMatch("9999.1", "Unknown", "9999", 0.5, "tfidf_retrieval")]

    svc = SnapshotService(Settings(), embedder=FakeEmbedder(), tfidf_matcher=NoMatchMatcher())
    resp = await svc.generate(_request(), session=object())

    assert resp.previous_occupation.role == "Project Coordinator"
    assert resp.previous_occupation.method == "embedding"


async def test_normalization_reaches_tier1():
    matcher = FakeMatcher()
    svc = SnapshotService(Settings(), embedder=FakeEmbedder(), tfidf_matcher=matcher)
    await svc.generate(_request(title="HR Manager"), session=object())

    # "HR Manager" is alias-normalized before Tier 1 sees it
    assert matcher.calls[0]["job_title"] == "human resources manager"


class FakeReranker:
    """Ranks by a fixed preference over role_ids; records the query it saw."""

    def __init__(self, preferred_order):
        self.preferred = preferred_order
        self.query_seen = None
        self.model_id = "fake-reranker"

    def rerank(self, query, candidates):
        from app.services.reranker import RerankResult

        self.query_seen = query

        def score(c):
            if c.role_id in self.preferred:
                return 1.0 - self.preferred.index(c.role_id) / 100
            return 0.01

        return sorted(
            (RerankResult(c.role_id, score(c)) for c in candidates),
            key=lambda r: r.score,
            reverse=True,
        )


async def test_reranker_reorders_recommendations_and_sets_method(monkeypatch):
    async def _texts(session, ids):
        labels = {
            "r_mkt": "Marketing Manager", "r1": "Project Coordinator",
            "r2": "Operations Executive", "r3": "Admin Executive", "r4": "HR Coordinator",
        }
        return {i: f"{labels.get(i, i)}. does stuff" for i in ids}

    monkeypatch.setattr(roles_repo, "get_rerank_texts", _texts)

    reranker = FakeReranker(preferred_order=["r2", "r_mkt"])
    svc = SnapshotService(
        Settings(), embedder=FakeEmbedder(), tfidf_matcher=FakeMatcher(), reranker=reranker
    )
    resp = await svc.generate(_request(), session=object())

    assert resp.previous_occupation.method == "reranker"
    assert resp.recommended_roles[0].role_id == "r2"
    assert resp.recommended_roles[0].role == "Operations Executive"
    assert 0.0 <= resp.recommended_roles[0].similarity <= 1.0
    assert reranker.query_seen and "skills:" in reranker.query_seen
    assert "Led project" not in reranker.query_seen  # raw CV text (PII surface) not sent


async def test_semantic_pass_reranks_candidates_by_definition(monkeypatch):
    # bi-encoder returns a near-neighbour (XQuery) ranked above the true skill (SQL)
    async def _two(session, vec, k, threshold):
        return [
            SkillMatch("s_xq", "XQuery", 0.70, "querying and transforming XML documents"),
            SkillMatch("s_sql", "SQL", 0.66, "retrieval of information from a database"),
        ]

    async def _texts(session, ids):
        return {i: f"role {i}" for i in ids}

    monkeypatch.setattr(skills_repo, "match_by_embedding", _two)
    monkeypatch.setattr(roles_repo, "get_rerank_texts", _texts)

    class PickSQL:
        model_id = "fake"

        def rerank(self, query, candidates):
            from app.services.reranker import RerankResult

            order = {"s_sql": 0.9, "s_xq": 0.1}
            return sorted(
                (RerankResult(c.role_id, order.get(c.role_id, 0.0)) for c in candidates),
                key=lambda r: r.score,
                reverse=True,
            )

    svc = SnapshotService(Settings(), embedder=FakeEmbedder(), tfidf_matcher=None, reranker=PickSQL())
    resp = await svc.generate(_request(), session=object())
    names = {p.skill for p in resp.professional_skills}
    assert "SQL" in names          # cross-encoder promoted the true skill
    assert "XQuery" not in names   # near-neighbour dropped: only top-1 per span kept


async def test_no_reranker_falls_back_to_existing_order():
    svc = SnapshotService(
        Settings(), embedder=FakeEmbedder(), tfidf_matcher=FakeMatcher(), reranker=None
    )
    resp = await svc.generate(_request(), session=object())
    assert resp.previous_occupation.method in ("classifier", "embedding")


async def test_reframe_dedupes():
    svc = SnapshotService(Settings(), embedder=None, tfidf_matcher=None)
    resp = await svc.generate(_request(), session=object())
    assert len(resp.reframed_skills) == 1
    assert resp.reframed_skills[0].skill == "Coordination"
    # no embedder -> no occupation match
    assert resp.previous_occupation is None
