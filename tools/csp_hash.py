"""Compute CSP script hashes for docs/*.html and check or update each page's policy.

Every inline <script> in a page must be allowed by a 'sha256-...' source in that
page's Content-Security-Policy meta tag. The hash covers the script's exact text,
comments and whitespace included, so any edit to a script needs a fresh hash.

Usage:
    python tools/csp_hash.py           check only; exits 1 if any page is out of date
    python tools/csp_hash.py --write   rewrite the script-src of out-of-date pages

Pages with no inline scripts must have no script-src at all (default-src 'none'
then blocks every script). External scripts (<script src=...>) are rejected:
the site doesn't use any, and privacy section 11.1 says so.
"""
import base64
import hashlib
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
SCRIPT_RE = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.S | re.I)
CSP_RE = re.compile(r'(<meta http-equiv="Content-Security-Policy" content=")(?P<policy>[^"]*)(">)')


def script_hashes(page: str) -> list[str]:
    """Return the CSP hash source for each inline script, in page order."""
    hashes = []
    for m in SCRIPT_RE.finditer(page):
        if "src=" in m.group("attrs").lower():
            raise ValueError("external script found; the site must not load script files")
        digest = hashlib.sha256(m.group("body").encode("utf-8")).digest()
        hashes.append("'sha256-" + base64.b64encode(digest).decode("ascii") + "'")
    return hashes


def expected_policy(policy: str, hashes: list[str]) -> str:
    """Return the policy with script-src replaced by exactly these hashes (or removed if none)."""
    parts = [p.strip() for p in policy.split(";") if p.strip() and not p.strip().startswith("script-src")]
    if hashes:
        parts.insert(1, "script-src " + " ".join(hashes))
    return "; ".join(parts)


def main() -> int:
    write = "--write" in sys.argv[1:]
    stale = 0
    for path in sorted(DOCS.glob("*.html")):
        text = path.read_text(encoding="utf-8")
        metas = list(CSP_RE.finditer(text))
        if len(metas) != 1:
            print(f"{path.name}: failed to find exactly one CSP meta tag (found {len(metas)})")
            return 2
        first_script = SCRIPT_RE.search(text)
        if first_script and first_script.start() < metas[0].start():
            print(f"{path.name}: a script comes before the CSP meta tag, so the policy would not cover it")
            return 2
        try:
            hashes = script_hashes(text)
        except ValueError as e:
            print(f"{path.name}: {e}")
            return 2
        old = metas[0].group("policy")
        new = expected_policy(old, hashes)
        if old == new:
            print(f"{path.name}: ok ({len(hashes)} script(s))")
            continue
        stale += 1
        if write:
            text = text[: metas[0].start("policy")] + new + text[metas[0].end("policy"):]
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"{path.name}: updated ({len(hashes)} script(s))")
        else:
            print(f"{path.name}: out of date; run with --write")
    return 1 if stale and not write else 0


if __name__ == "__main__":
    sys.exit(main())
