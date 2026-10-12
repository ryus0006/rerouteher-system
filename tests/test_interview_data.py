"""E7 dataset and migration validation.

Reads the authoritative CSVs and the migration SQL directly from disk. No database
connection is used here: the actual COPY/INSERT behaviour against a live Postgres is
exercised separately once a real environment is available (see docs/tasks for the
database-initialization verification boundary).
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "db"

QUESTION_CSV = DB_DIR / "interview_question.csv"
RUBRIC_CSV = DB_DIR / "e7_interview_evaluation_rubric.csv"
MAPPING_CSV = DB_DIR / "e7_interview_question_rubric.csv"
MIGRATION_SQL = DB_DIR / "07_e7_interview_coach.sql"

MOJIBAKE_SEQUENCES = ["â€œ", "â€\x9d", "â€™", "â€˜"]

CONTEXT_COLUMNS = [
    "interview_method_sources",
    "authoring_method",
    "answer_framework",
    "answer_guidance",
    "strong_evidence_signals",
    "watch_out_for",
    "follow_up_question",
]


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_question_csv_is_valid_utf8_with_no_mojibake():
    text = QUESTION_CSV.read_text(encoding="utf-8")
    for seq in MOJIBAKE_SEQUENCES:
        assert seq not in text, f"mojibake sequence {seq!r} still present"


def test_question_and_rubric_json_parse_for_every_row():
    for row in _read_rows(QUESTION_CSV):
        anchor = json.loads(row["anchor_json"])
        rubric = json.loads(row["rubric_json"])
        assert isinstance(anchor, dict)
        assert rubric["source"] == "e7_interview_evaluation_rubric.csv"


def test_question_ids_are_unique():
    rows = _read_rows(QUESTION_CSV)
    ids = [r["question_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_criterion_ids_are_unique():
    rows = _read_rows(RUBRIC_CSV)
    ids = [r["criterion_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_mapping_pairs_are_unique():
    rows = _read_rows(MAPPING_CSV)
    pairs = [(r["question_id"], r["criterion_id"]) for r in rows]
    assert len(pairs) == len(set(pairs))


def test_exactly_20_general_questions():
    rows = _read_rows(QUESTION_CSV)
    general = [r for r in rows if r["role_id"] == "NULL"]
    assert len(general) == 20


def test_631_role_ids_have_exactly_12_role_specific_questions_each():
    rows = _read_rows(QUESTION_CSV)
    role_rows = [r for r in rows if r["role_id"] != "NULL"]
    counts = Counter(r["role_id"] for r in role_rows)
    assert len(counts) == 631
    assert all(count == 12 for count in counts.values())


def test_all_mapping_foreign_keys_resolve():
    question_ids = {r["question_id"] for r in _read_rows(QUESTION_CSV)}
    criterion_ids = {r["criterion_id"] for r in _read_rows(RUBRIC_CSV)}
    for row in _read_rows(MAPPING_CSV):
        assert row["question_id"] in question_ids
        assert row["criterion_id"] in criterion_ids


def test_every_question_maps_to_eval_01_05_and_10():
    mapping = _read_rows(MAPPING_CSV)
    by_question: dict[str, set[str]] = {}
    for row in mapping:
        by_question.setdefault(row["question_id"], set()).add(row["criterion_id"])
    question_ids = {r["question_id"] for r in _read_rows(QUESTION_CSV)}
    required = {"EVAL-01", "EVAL-05", "EVAL-10"}
    for qid in question_ids:
        assert required.issubset(by_question.get(qid, set())), qid


def test_every_question_has_between_3_and_8_criteria():
    mapping = _read_rows(MAPPING_CSV)
    counts = Counter(row["question_id"] for row in mapping)
    assert all(3 <= count <= 8 for count in counts.values())


def test_counts_match_the_migrations_asserted_totals():
    assert len(_read_rows(QUESTION_CSV)) == 7592
    assert len(_read_rows(RUBRIC_CSV)) == 10
    assert len(_read_rows(MAPPING_CSV)) == 44902


def test_migration_creates_only_the_five_interview_tables():
    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    created = re.findall(
        r"CREATE TABLE IF NOT EXISTS rerouteher\.(\w+)", sql
    )
    assert set(created) == {
        "ai_evaluation_rubric",
        "interview_question",
        "interview_question_rubric",
        "interview_session",
        "interview_response",
    }


def test_migration_asserts_staging_counts_before_insert():
    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    assert "7592" in sql
    assert "44902" in sql
    assert "RAISE EXCEPTION" in sql


def test_migration_copies_from_the_mounted_csv_paths():
    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    for filename in ("e7_interview_evaluation_rubric.csv", "interview_question.csv", "e7_interview_question_rubric.csv"):
        assert f"/docker-entrypoint-initdb.d/{filename}" in sql


def test_question_csv_has_all_seven_context_columns():
    rows = _read_rows(QUESTION_CSV)
    assert set(CONTEXT_COLUMNS).issubset(rows[0].keys())


def test_every_question_has_non_empty_context_in_every_column():
    for row in _read_rows(QUESTION_CSV):
        for col in CONTEXT_COLUMNS:
            assert row[col].strip(), f"{row['question_id']} missing {col}"


def test_migration_stores_the_seven_context_columns():
    sql = MIGRATION_SQL.read_text(encoding="utf-8")
    for col in CONTEXT_COLUMNS:
        assert col in sql
