# Test fixtures

Reusable sample CV PDFs so tests and smoke checks do not rebuild a CV each time.
Regenerate with `python3 tests/fixtures/generate_fixtures.py`.

| File | CV | Exercises |
|---|---|---|
| `cv_marketing_executive.pdf` | Marketing Executive (curated role) | occupation via MASCO lookup (Tier 1 classifier); PII redaction |
| `cv_bookkeeper.pdf` | Bookkeeper (curated role) | occupation via MASCO lookup (Tier 1 classifier) |
| `cv_software_engineer.pdf` | Software Engineer | skill extraction + title normalization (software engineer -> software developer) |
| `cv_registered_nurse.pdf` | Registered Nurse (outside curated roles) | occupation embedding fallback (Tier 2) |
| `cv_two_column_ux.pdf` | UX Designer, two-column layout | column-aware reading order (skills left, experience right) |

Load one in a test with:

```python
from pathlib import Path
FIXTURES = Path(__file__).parent / "fixtures"
pdf_bytes = (FIXTURES / "cv_marketing_executive.pdf").read_bytes()
```
