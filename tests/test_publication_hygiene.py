"""Observable safeguards for the publication audit; no real credentials used."""

from tools.check_publication import check_paths


def test_hygiene_reports_a_path_without_echoing_a_credential(tmp_path):
    path = tmp_path / "sample.md"
    fake = "gh" + "p_" + "x" * 36
    path.write_text(fake, encoding="utf-8")
    assert check_paths([path], tmp_path) == [("sample.md", "github-token")]
    assert fake not in str(check_paths([path], tmp_path))


def test_hygiene_never_reads_environment_files(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"\xff")
    assert check_paths([path], tmp_path) == [(".env", "environment-file")]


def test_hygiene_checks_local_links_and_allows_external_or_anchor_links(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("[local](missing.md) [web](https://example.com) [anchor](#top)")
    assert check_paths([path], tmp_path) == [("README.md", "broken-local-link")]
