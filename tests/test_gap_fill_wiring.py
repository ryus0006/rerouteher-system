from types import SimpleNamespace

from app.api.gap import compute_gap
from app.schemas.gap import Gap, GapRequest, GapResponse


class FakeGapService:
    async def compute(self, req, session):
        return GapResponse(readiness=0.5, gaps=[
            Gap(skill_id="s1", skill="SQL", band="role", importance=80.0, uplift=0.2),
            Gap(skill_id="s2", skill="Excel", band="role", importance=70.0, uplift=0.1),
        ])


class FakeFill:
    def __init__(self, enabled=True):
        self.enabled = enabled

    async def fill_for_skills(self, skill_ids):
        pass


class FakeBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args):
        self.tasks.append((fn, args))


def _request(**state):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


async def test_compute_gap_schedules_fill_with_gap_skill_ids():
    fill = FakeFill(enabled=True)
    bg = FakeBackgroundTasks()
    req = GapRequest(skill_ids=[], target_role_id="r1")
    resp = await compute_gap(req, _request(gap_service=FakeGapService(), learning_fill_service=fill), bg, session=None)
    assert resp.readiness == 0.5
    assert len(bg.tasks) == 1
    fn, args = bg.tasks[0]
    assert fn == fill.fill_for_skills and args == (["s1", "s2"],)


async def test_compute_gap_skips_when_fill_disabled():
    bg = FakeBackgroundTasks()
    req = GapRequest(skill_ids=[], target_role_id="r1")
    await compute_gap(req, _request(gap_service=FakeGapService(), learning_fill_service=FakeFill(enabled=False)), bg, session=None)
    assert bg.tasks == []


async def test_compute_gap_handles_missing_fill_service():
    bg = FakeBackgroundTasks()
    req = GapRequest(skill_ids=[], target_role_id="r1")
    resp = await compute_gap(req, _request(gap_service=FakeGapService()), bg, session=None)
    assert resp.readiness == 0.5 and bg.tasks == []


class BigGapService:
    """Returns many gaps (like a real role) to prove the fill is capped to the
    learning page's focus areas, not the whole list."""
    async def compute(self, req, session):
        gaps = [Gap(skill_id=f"a{i}", skill="AI", band="ai_usage", importance=50.0, uplift=0.9 - i * 0.01)
                for i in range(5)]
        gaps += [Gap(skill_id=f"r{i}", skill="Role", band="role", importance=80.0, uplift=0.8 - i * 0.01)
                 for i in range(20)]
        return GapResponse(readiness=0.1, gaps=gaps)


async def test_compute_gap_caps_fill_to_focus_areas():
    fill = FakeFill(enabled=True)
    bg = FakeBackgroundTasks()
    req = GapRequest(skill_ids=[], target_role_id="r1")
    await compute_gap(req, _request(gap_service=BigGapService(), learning_fill_service=fill), bg, session=None)
    assert len(bg.tasks) == 1
    _fn, args = bg.tasks[0]
    scheduled = args[0]
    # exactly 3 (MAX_FOCUS_AREAS): 2 top role gaps + 1 top ai gap, never all 25.
    # Ordered by uplift desc, so the ai gap (0.9) leads the role gaps (0.8, 0.79).
    assert len(scheduled) == 3
    assert scheduled == ["a0", "r0", "r1"]
