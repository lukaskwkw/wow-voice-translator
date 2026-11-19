"""set_secret writes keys to .env without touching other lines (no network)."""

import os

import pytest

from wvt.envfile import set_secret

NAME = "WVT_TEST_SECRET_KEY"


def _cleanup() -> None:
    os.environ.pop(NAME, None)


def test_creates_env_file(tmp_path):
    path = tmp_path / ".env"
    try:
        set_secret(NAME, "s3cr3t", path=path)
    finally:
        _cleanup()
    assert path.is_file()
    assert path.read_text(encoding="utf-8").strip() == f"{NAME}=s3cr3t"


def test_appends_new_key_and_preserves_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text("# comment\nOTHER=1\n", encoding="utf-8")
    try:
        set_secret(NAME, "abc", path=path)
    finally:
        _cleanup()
    text = path.read_text(encoding="utf-8")
    assert "# comment" in text
    assert "OTHER=1" in text
    assert f"{NAME}=abc" in text


def test_updates_existing_key(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f"{NAME}=old\nOTHER=2\n", encoding="utf-8")
    try:
        set_secret(NAME, "new", path=path)
    finally:
        _cleanup()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert f"{NAME}=new" in lines
    assert "OTHER=2" in lines
    assert not any("old" in line for line in lines)


def test_updates_export_line(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f"# keep\nexport {NAME}=old\n", encoding="utf-8")
    try:
        set_secret(NAME, "fresh", path=path)
    finally:
        _cleanup()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# keep"
    assert f"export {NAME}=fresh" in lines
    assert not any("old" in line for line in lines)


def test_sets_os_environ(tmp_path):
    path = tmp_path / ".env"
    try:
        set_secret(NAME, "value-42", path=path)
        assert os.environ[NAME] == "value-42"
    finally:
        _cleanup()


@pytest.mark.parametrize("bad", ["", "1BAD", "BAD-NAME", "bad name", "BAD\nNAME"])
def test_rejects_bad_name(tmp_path, bad):
    with pytest.raises(ValueError):
        set_secret(bad, "x", path=tmp_path / ".env")


def test_rejects_newline_value(tmp_path):
    with pytest.raises(ValueError):
        set_secret(NAME, "line1\nline2", path=tmp_path / ".env")
