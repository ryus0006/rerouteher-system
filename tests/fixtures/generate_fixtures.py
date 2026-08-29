"""Generate the sample CV PDFs used across tests and smoke checks.

Run from anywhere:  python3 tests/fixtures/generate_fixtures.py
Writes the .pdf files next to this script. Commit the PDFs so tests can reuse them
without regenerating.
"""
from pathlib import Path

import pymupdf

HERE = Path(__file__).parent

# name -> CV plain text. Single-column layout.
CVS = {
    "cv_marketing_executive": (
        "Aisha Rahman\n"
        "aisha.rahman@example.com | +60 12-987 6543\n"
        "\n"
        "Work Experience\n"
        "Marketing Executive\n"
        "Nestle Malaysia\n"
        "Jan 2015 - Dec 2019\n"
        "Managed digital marketing campaigns, social media, and stakeholder management.\n"
        "Led budgeting and content writing for product launches.\n"
        "\n"
        "Education\n"
        "BBA Marketing, 2014\n"
    ),
    "cv_bookkeeper": (
        "Mei Ling Tan\n"
        "\n"
        "Work Experience\n"
        "Bookkeeper\n"
        "Sunrise Trading Sdn Bhd\n"
        "Mar 2016 - Aug 2021\n"
        "Recorded transactions, reconciled accounts, processed payroll and invoices.\n"
        "Prepared monthly financial reports and managed accounts payable.\n"
        "\n"
        "Education\n"
        "Diploma in Accounting, 2015\n"
    ),
    "cv_software_engineer": (
        "Arjun Kumar\n"
        "\n"
        "Professional Experience\n"
        "Software Engineer\n"
        "Grab Malaysia\n"
        "Feb 2018 - Present\n"
        "Built Python REST APIs, wrote automated tests, and worked with SQL databases.\n"
        "\n"
        "Skills\n"
        "Python, SQL, REST API, software testing, Git\n"
        "\n"
        "Education\n"
        "BSc Computer Science, 2017\n"
    ),
    "cv_registered_nurse": (
        "Siti Aminah\n"
        "\n"
        "Work Experience\n"
        "Registered Nurse\n"
        "Hospital Kuala Lumpur\n"
        "Jan 2014 - Dec 2020\n"
        "Provided patient care, administered medication, monitored vital signs, and assisted in surgery.\n"
        "\n"
        "Education\n"
        "Diploma in Nursing, 2013\n"
    ),
}


def write_single_column(text: str, path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(path)
    doc.close()


def write_two_column(path: Path) -> None:
    """A two-column CV (skills left, experience right) to exercise column reading order."""
    left = (
        "SKILLS\n"
        "User research\n"
        "Prototyping\n"
        "Figma\n"
        "Wireframing\n"
        "\n"
        "LANGUAGES\n"
        "English\n"
        "Malay\n"
    )
    right = (
        "EXPERIENCE\n"
        "UX Designer\n"
        "Fave Malaysia\n"
        "Jan 2017 - Dec 2021\n"
        "Ran user research, built prototypes in Figma, and led design reviews.\n"
        "\n"
        "EDUCATION\n"
        "BA Design, 2016\n"
    )
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), left, fontsize=11)      # left column
    page.insert_text((330, 72), right, fontsize=11)    # right column
    doc.save(path)
    doc.close()


def main() -> None:
    for name, text in CVS.items():
        out = HERE / f"{name}.pdf"
        write_single_column(text, out)
        print("wrote", out.name)
    two_col = HERE / "cv_two_column_ux.pdf"
    write_two_column(two_col)
    print("wrote", two_col.name)


if __name__ == "__main__":
    main()
