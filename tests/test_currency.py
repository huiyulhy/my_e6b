"""Data currency: the dates come off disk, not out of a constant."""

from datetime import date

import pytest

from engine import currency as cur


@pytest.fixture
def readme(tmp_path):
    def write(first_line: str):
        path = tmp_path / "README.txt"
        path.write_text(f"{first_line}\n\nDear Subscribers,\n")
        return path

    return write


def test_reads_the_effective_date_from_the_first_line(readme):
    path = readme("AIS subscriber files effective date August 06, 2026.")
    assert cur.nasr_effective_date(path) == date(2026, 8, 6)


def test_missing_readme_is_not_an_error(tmp_path):
    assert cur.nasr_effective_date(tmp_path / "nope.txt") is None


def test_unparseable_first_line_is_not_an_error(readme):
    assert cur.nasr_effective_date(readme("Dear Subscribers,")) is None


def test_nasr_expires_one_month_after_the_effective_date(readme, tmp_path):
    data = cur.nasr(readme_path=readme("effective date January 31, 2026."),
                    db_path=tmp_path / "none.sqlite")
    assert data.effective == date(2026, 1, 31)
    # February has no 31st; clamping to the month end is the safe direction.
    assert data.expires == date(2026, 2, 28)
    assert not data.expired(date(2026, 2, 28))
    assert data.expired(date(2026, 3, 1))
    assert data.days_remaining(date(2026, 2, 20)) == 8


def test_bundled_nasr_readme_parses():
    assert cur.nasr_effective_date() is not None


def test_wmm_window_is_five_years_from_the_epoch():
    data = cur.world_magnetic_model()
    assert data.effective == date(2025, 1, 1)
    assert data.expires == date(2030, 1, 1)
    assert not data.expired(date(2029, 12, 31))
    assert data.expired(date(2030, 1, 2))


def test_stale_airport_database_is_flagged(readme, tmp_path):
    import sqlite3

    db = tmp_path / "airports.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
    connection.execute("INSERT INTO metadata VALUES ('effective_date', '2026-06-11')")
    connection.commit()
    connection.close()

    data = cur.nasr(readme_path=readme("effective date August 06, 2026."), db_path=db)
    assert data.effective == date(2026, 8, 6)
    assert data.note and "2026-06-11" in data.note
