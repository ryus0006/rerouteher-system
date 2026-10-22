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
from app.services.snapshot import SnapshotService, _phrase_set


def test_phrase_set_ignores_pii_redaction_markers():
    phrases = _phrase_set("built apis using [email] and sql, redacted [phone]")
    # The inner words of redaction markers must not become skill phrases.
    assert "email" not in phrases and "phone" not in phrases
    assert "sql" in phrases

pytestmark = pytest.mark.asyncio

SKILL_ROWS = [
    SkillRow("s_pm", "Project management", "soft", "Planning and coordinating work."),
    SkillRow("s_budget", "Budgeting", "soft", None),
    SkillRow("s_ux", "User research", "technical", "Understanding users and their needs."),
    SkillRow("s_coord", "Coordination", "soft", "Bringing people and tasks together."),
]

# (skill_id, term) pairs: canonical names + aliases, as load_alias_dictionary returns
ALIAS_PAIRS = [
    ("s_pm", "Project management"),
    ("s_pm", "project coordination"),
    ("s_budget", "Budgeting"),
    ("s_budget", "budget management"),
    ("s_ux", "User research"),
    ("s_ux", "ux research"),
    ("s_coord", "Coordination"),
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

    async def _nearest(session, vec, k, exclude=()):
        roles = [
            NearestRole("r1", "Project Coordinator", 0.91),
            NearestRole("r2", "Operations Executive", 0.77),
            NearestRole("r3", "Admin Executive", 0.66),
            NearestRole("r4", "HR Coordinator", 0.60),
        ]
        return [r for r in roles if r.role_id not in exclude]

    async def _get_by_esco(session, code, esco_title=None, exclude=()):
        # the FakeMatcher's esco_code resolves to a role
        if code == "1234.1" and "r_mkt" not in exclude:
            return Role("r_mkt", "Marketing Manager", "1234")
        return None

    async def _get_by_masco(session, code, exclude=()):
        if code == "1234" and "r_mkt" not in exclude:
            return Role("r_mkt", "Marketing Manager", "1234")
        return None

    async def _placeholders(session, min_shared):
        return set()

    monkeypatch.setattr(skills_repo, "list_skills", _list_skills)
    monkeypatch.setattr(skills_repo, "load_alias_dictionary", _load_alias_dictionary)
    monkeypatch.setattr(skills_repo, "match_by_embedding", _match_by_embedding)
    monkeypatch.setattr(caregiving_repo, "reframe", _reframe)
    monkeypatch.setattr(roles_repo, "nearest_by_embedding", _nearest)
    monkeypatch.setattr(roles_repo, "get_by_esco_code", _get_by_esco)
    monkeypatch.setattr(roles_repo, "get_by_masco_code", _get_by_masco)
    monkeypatch.setattr(roles_repo, "placeholder_role_ids", _placeholders)


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
    definitions = {p.skill: p.definition for p in resp.professional_skills}
    assert definitions["Project management"] == "Planning and coordinating work."
    assert definitions["User research"] == "Understanding users and their needs."
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
    assert resp.reframed_skills[0].skill_id == "s_coord"
    assert resp.reframed_skills[0].definition == "Bringing people and tasks together."
    # no embedder -> no occupation match
    assert resp.previous_occupation is None


async def test_resolve_role_uses_embedding_nearest_no_reranker():
    # patch_repos.nearest_by_embedding returns Project Coordinator first; no reranker to reorder
    svc = SnapshotService(Settings(), embedder=FakeEmbedder(), tfidf_matcher=None, reranker=None)
    role = await svc.resolve_role("project coordinator", [], session=object())
    assert role.role_id == "r1" and role.role_title == "Project Coordinator"


async def test_resolve_role_reranker_reorders(monkeypatch):
    async def _texts(session, ids):
        return {rid: f"text {rid}" for rid in ids}

    monkeypatch.setattr(roles_repo, "get_rerank_texts", _texts)

    class PickR2:
        model_id = "fake"

        def rerank(self, query, candidates):
            from app.services.reranker import RerankResult

            # rank r2 top regardless of embedding order
            return sorted(
                (RerankResult(c.role_id, 1.0 if c.role_id == "r2" else 0.1) for c in candidates),
                key=lambda r: r.score,
                reverse=True,
            )

    svc = SnapshotService(
        Settings(), embedder=FakeEmbedder(), tfidf_matcher=None, reranker=PickR2()
    )
    role = await svc.resolve_role("operations", [], session=object())
    assert role.role_id == "r2"  # reranker pick, not the nearest-by-embedding top


async def test_resolve_role_none_when_no_embedder_and_no_tfidf():
    svc = SnapshotService(Settings(), embedder=None, tfidf_matcher=None, reranker=None)
    assert await svc.resolve_role("marketing manager", [], session=object()) is None


async def test_confirmed_skills_merge_into_professional():
    svc = SnapshotService(Settings(), embedder=None, tfidf_matcher=None, reranker=None)
    cv = CV(raw_text="", experiences=[], skill_mentions=[])
    req = SnapshotRequest(
        cv=cv, break_=Break(duration_years=0, activities=[]), confirmed_skills=["s_ux"]
    )
    resp = await svc.generate(req, session=object())
    confirmed = [p for p in resp.professional_skills if p.source == "role_confirmed"]
    assert any(
        p.skill_id == "s_ux"
        and p.skill == "User research"
        and p.definition == "Understanding users and their needs."
        for p in confirmed
    )


async def test_missing_skill_definition_is_null():
    svc = SnapshotService(Settings(), embedder=None, tfidf_matcher=None, reranker=None)
    req = SnapshotRequest(
        cv=CV(raw_text="budgeting", experiences=[], skill_mentions=["budgeting"]),
        break_=Break(duration_years=0, activities=[]),
    )
    resp = await svc.generate(req, session=object())
    budget = next(p for p in resp.professional_skills if p.skill_id == "s_budget")
    assert budget.definition is None


async def test_confirmed_skill_does_not_override_cv_evidence():
    # s_budget is extracted from the CV ("budgeting" mention); confirming it must not
    # downgrade its source to role_confirmed.
    svc = SnapshotService(Settings(), embedder=None, tfidf_matcher=None, reranker=None)
    req = SnapshotRequest(
        cv=CV(raw_text="budgeting", experiences=[], skill_mentions=["budgeting"]),
        break_=Break(duration_years=0, activities=[]),
        confirmed_skills=["s_budget"],
    )
    resp = await svc.generate(req, session=object())
    budget = next(p for p in resp.professional_skills if p.skill_id == "s_budget")
    assert budget.source == "experience"


def _ranked(*role_ids):
    from app.services.reranker import RerankResult
    return [RerankResult(rid, 1.0 - i * 0.1) for i, rid in enumerate(role_ids)]


def test_recommendations_after_the_top_pick_put_same_job_group_first():
    # top pick is in MASCO group 251; the reranker's 2nd/3rd are data roles (252), which
    # follow the same-group roles, each part keeping the reranker order
    pool = {
        "A": Role("A", "Software Developer", "251201"),
        "B": NearestRole("B", "Data Analyst", 0.5, "252403"),
        "C": NearestRole("C", "Database Administrator", 0.5, "252103"),
        "D": NearestRole("D", "Systems Programmer", 0.5, "251407"),
        "E": NearestRole("E", "Website Developer", 0.5, "251301"),
    }
    ranked = _ranked("A", "B", "C", "D", "E")
    ordered = SnapshotService._same_group_first(ranked[0], ranked[1:], pool)
    assert [x.role_id for x in ordered] == ["D", "E", "B", "C"]


def test_same_group_first_keeps_reranker_order_without_a_group():
    pool = {"A": NearestRole("A", "Top", 0.9), "B": NearestRole("B", "Two", 0.5, "251201"),
            "C": NearestRole("C", "Three", 0.5, "251401")}
    ranked = _ranked("A", "B", "C")
    assert [x.role_id for x in SnapshotService._same_group_first(ranked[0], ranked[1:], pool)] == ["B", "C"]


async def test_phrase_pass_adds_unknown_mentions_but_keeps_them_out_of_the_role_query(monkeypatch):
    # "digital marketing" has no alias entry and its sentence does not mention it, so only
    # the phrase pass finds it: it joins her skills (evidence = her phrase) but not the query
    class MarkingEmbedder(FakeEmbedder):
        def encode(self, texts):
            out = np.zeros((len(texts), 384), dtype="float32")
            for i, t in enumerate(texts):
                out[i, 0] = 1.0 if "digital marketing" in t else 0.0
            return out

    async def _match(session, vec, k, threshold):
        if vec[0] == 1.0:
            return [SkillMatch("s_dm", "Digital Marketing Techniques", 0.76, "promoting online")]
        return []

    async def _list_skills(session):
        return SKILL_ROWS + [SkillRow("s_dm", "Digital Marketing Techniques", "technical", "promoting online")]

    async def _texts(session, ids):
        return {i: f"role {i}" for i in ids}

    monkeypatch.setattr(skills_repo, "match_by_embedding", _match)
    monkeypatch.setattr(skills_repo, "list_skills", _list_skills)
    monkeypatch.setattr(roles_repo, "get_rerank_texts", _texts)
    reranker = FakeReranker(preferred_order=["r1"])
    svc = SnapshotService(Settings(), embedder=MarkingEmbedder(), tfidf_matcher=None, reranker=reranker)
    req = SnapshotRequest(
        cv=CV(
            raw_text="Ran campaigns and budgeting.",
            experiences=[Experience(title="Executive", description="Ran campaigns.")],
            skill_mentions=["digital marketing", "budgeting"],
        ),
        break_=Break(duration_years=0, activities=[]),
    )
    resp = await svc.generate(req, session=object())

    dm = next(p for p in resp.professional_skills if p.skill_id == "s_dm")
    assert dm.skill == "Digital Marketing Techniques" and dm.evidence == "digital marketing"
    assert "budgeting" in reranker.query_seen  # alias-matched skills still inform the role
    assert "digital marketing" not in reranker.query_seen


async def test_placeholder_roles_are_left_out_of_every_role_search(monkeypatch):
    # r_mkt (Tier 1) and r1 (nearest) sit on a catch-all placeholder skill list
    seen = []

    async def _placeholders(session, min_shared):
        seen.append(min_shared)
        return {"r_mkt", "r1"}

    async def _texts(session, ids):
        return {i: f"role {i}" for i in ids}

    monkeypatch.setattr(roles_repo, "placeholder_role_ids", _placeholders)
    monkeypatch.setattr(roles_repo, "get_rerank_texts", _texts)
    svc = SnapshotService(
        Settings(), embedder=FakeEmbedder(), tfidf_matcher=FakeMatcher(),
        reranker=FakeReranker(preferred_order=["r_mkt", "r1", "r2"]),
    )
    resp = await svc.generate(_request(), session=object())
    ids = [r.role_id for r in resp.recommended_roles]
    assert "r_mkt" not in ids and "r1" not in ids
    assert ids[0] == "r2"
    assert seen == [Settings().placeholder_role_min_shared]  # looked up once
    await svc.generate(_request(), session=object())
    assert len(seen) == 1  # cached on the service


async def test_resolve_role_skips_placeholder_roles(monkeypatch):
    async def _placeholders(session, min_shared):
        return {"r1"}

    monkeypatch.setattr(roles_repo, "placeholder_role_ids", _placeholders)
    svc = SnapshotService(Settings(), embedder=FakeEmbedder(), tfidf_matcher=None, reranker=None)
    role = await svc.resolve_role("Project Coordinator", [], session=object())
    assert role.role_id == "r2"  # nearest non-placeholder role


def test_job_group_order_falls_back_to_the_two_digit_group():
    # top pick Chemist (211); nothing else in 211, so the 21 roles come before the rest
    pool = {
        "A": Role("A", "Chemist", "211302"),
        "B": NearestRole("B", "QA Engineering Technician", 0.5, "311928"),
        "C": NearestRole("C", "Chemical Engineer", 0.5, "214503"),
        "D": NearestRole("D", "Chemistry Technician", 0.5, "311106"),
        "E": NearestRole("E", "Bio-Fermentation Specialist", 0.5, "213433"),
    }
    ranked = _ranked("A", "B", "C", "D", "E")
    ordered = SnapshotService._same_group_first(ranked[0], ranked[1:], pool)
    assert [x.role_id for x in ordered] == ["C", "E", "B", "D"]


async def test_top_pick_is_the_role_linked_to_two_of_her_titles(monkeypatch):
    # the classifier links r_mkt to both titles; the reranker prefers r2, which only the
    # embedding search found, so r_mkt still becomes her previous occupation
    async def _texts(session, ids):
        return {i: f"role {i}" for i in ids}

    monkeypatch.setattr(roles_repo, "get_rerank_texts", _texts)
    svc = SnapshotService(
        Settings(), embedder=FakeEmbedder(), tfidf_matcher=FakeMatcher(),
        reranker=FakeReranker(preferred_order=["r2", "r_mkt"]),
    )
    req = SnapshotRequest(
        cv=CV(
            raw_text="Quality assurance chemist. Research chemist.",
            experiences=[Experience(title="Quality Assurance Chemist"), Experience(title="Research Chemist")],
            skill_mentions=[],
        ),
        break_=Break(duration_years=0, activities=[]),
    )
    resp = await svc.generate(req, session=object())
    assert resp.previous_occupation.role_id == "r_mkt"
    assert [r.role_id for r in resp.recommended_roles][:2] == ["r_mkt", "r2"]
