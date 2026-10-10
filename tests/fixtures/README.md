# Test fixtures

Reusable sample CV PDFs so tests and smoke checks do not rebuild a CV each time.

Minimal fixtures below exercise specific parser behaviours; regenerate them with
`python3 tests/fixtures/generate_fixtures.py`.

| File | CV | Exercises |
|---|---|---|
| `cv_marketing_executive.pdf` | Marketing Executive (curated role) | occupation via MASCO lookup (Tier 1 classifier); PII redaction |
| `cv_bookkeeper.pdf` | Bookkeeper (curated role) | occupation via MASCO lookup (Tier 1 classifier) |
| `cv_software_engineer.pdf` | Software Engineer | skill extraction + title normalization (software engineer -> software developer) |
| `cv_registered_nurse.pdf` | Registered Nurse (outside curated roles) | occupation embedding fallback (Tier 2) |
| `cv_two_column_ux.pdf` | UX Designer, two-column layout | column-aware reading order (skills left, experience right) |

Complete, themed STEM sample CVs below are committed source PDFs (not produced by the
generator). Each has a STEM latest role and all-STEM history that maps to the STEM roles the
production DB carries, with a full header, summary, competencies and experience section.

| File | CV |
|---|---|
| `cv_software_developer.pdf` | Senior Software Developer (software/IT) |
| `cv_data_analyst.pdf` | Data Analyst (data/analytics) |
| `cv_chemist.pdf` | Quality Assurance Chemist (science/lab) |
| `cv_mechanical_engineer.pdf` | Mechanical Engineer (engineering) |

Load one in a test with:

```python
from pathlib import Path
FIXTURES = Path(__file__).parent / "fixtures"
pdf_bytes = (FIXTURES / "cv_marketing_executive.pdf").read_bytes()
```
