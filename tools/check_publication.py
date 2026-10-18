"""Check publication files for broken local links and common credential literals.

Uses Git's publication candidate list, so ignored environments/scratch outputs
are excluded. Matches report only the path/rule, never credential contents.
This is a lightweight hygiene check, not a complete secret-scanning service.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".ipynb", ".yml", ".yaml", ".txt",
                 ".toml", ".ps1", ".svg", ""}
SECRET_PATTERNS = {
    "private-key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "openai-key": re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b"),
    "aws-access-key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
}


def candidate_paths(root: Path = ROOT):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root, check=True, capture_output=True,
    )
    return sorted({root / name.decode("utf-8") for name in result.stdout.split(b"\0") if name})


def check_paths(paths, root: Path = ROOT):
    findings = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        name = path.name.lower()
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            findings.append((relative, "environment-file"))
            continue  # Never open a real environment/credential file.
        if name in {".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519"}:
            findings.append((relative, "credential-file"))
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeError, OSError):
            findings.append((relative, "unreadable-text"))
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append((relative, name))
        if path.suffix.lower() != ".md":
            continue
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = unquote(target.strip().strip("<>").split("#", 1)[0])
            if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            destination = (path.parent / target).resolve()
            if not destination.is_relative_to(root.resolve()) or not destination.exists():
                findings.append((relative, "broken-local-link"))
    return findings


def main():
    paths = candidate_paths()
    findings = check_paths(paths)
    for path, rule in findings:
        print(f"{path}: {rule}")
    print(f"checked {len(paths)} publication candidates; {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
