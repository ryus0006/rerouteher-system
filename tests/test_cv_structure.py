"""Tests for the LLM CV structuring service (hybrid parse)."""
import pytest

from app.services.cv_structure import CvStructureService
from app.services.llm import GenerateResult, LlmError

pytestmark = pytest.mark.asyncio


class _Llm:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.seen_text = None

    async def generate(self, *, system_instruction, contents, tools):
        self.seen_text = contents[0]["parts"][0]["text"]
        if self._error is not None:
            raise self._error
        return self._result


def _submit(experiences, skills):
    return GenerateResult(
        content={
            "parts": [
                {"functionCall": {"name": "submit_cv", "args": {"experiences": experiences, "skills": skills}}}
            ]
        }
    )


async def test_structures_experiences_and_skills_from_masked_text():
    llm = _Llm(
        _submit(
            [
                {
                    "title": "Marketing Executive",
                    "organisation": "Klang Valley Media Sdn Bhd",
                    "start": "Jan 2010",
                    "end": "Jan 2013",
                    "description": "Supported marketing campaigns and events.",
                }
            ],
            ["digital marketing", "SQL", "SQL", "digital marketing"],
        )
    )
    service = CvStructureService(llm=llm)
    result = await service.structure("N**** A***** \nMarketing Executive\n...")

    assert result is not None
    experiences, skills = result
    assert llm.seen_text.startswith("N****")  # the masked text was sent, not raw PII
    assert len(experiences) == 1
    assert experiences[0].title == "Marketing Executive"
    assert experiences[0].organisation == "Klang Valley Media Sdn Bhd"
    assert experiences[0].start == "Jan 2010"
    assert skills == ["digital marketing", "SQL"]  # de-duplicated, order preserved


async def test_drops_entries_without_title_or_organisation():
    llm = _Llm(_submit([{"start": "2019", "end": "2020"}, {"title": "Analyst"}], []))
    experiences, _ = await CvStructureService(llm=llm).structure("text")
    assert [e.title for e in experiences] == ["Analyst"]


async def test_returns_none_without_llm():
    assert await CvStructureService(llm=None).structure("text") is None


async def test_returns_none_on_llm_error():
    service = CvStructureService(llm=_Llm(error=LlmError("boom")))
    assert await service.structure("text") is None


async def test_returns_none_when_no_tool_call():
    service = CvStructureService(llm=_Llm(GenerateResult(content={"parts": [{"text": "hi"}]})))
    assert await service.structure("text") is None


async def test_returns_none_for_empty_text():
    assert await CvStructureService(llm=_Llm(_submit([], []))).structure("   ") is None
