"""Link integrity for the live documentation set (change `docs-split-large-docs`).

Splitting a large document produces a diff nobody can review line by line. What
makes the split safe is not the review — it is this test: every relative
Markdown link in the live tree must resolve to a file that exists, and every
`#anchor` must match a heading in that file.

Scope is the *live* documentation. `openspec/changes/archive/**` and
`docs/superpowers/**` record what was true when they landed, so they keep the
links they were written with and are excluded by design. Absolute `http(s)`
links are not checked — the test box has no network.
"""

from __future__ import annotations

import os
import re
from collections import Counter

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Root-level documents that are part of the live set.
ROOT_DOCS = (
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
    "CHANGELOG.md",
    os.path.join("examples", "README.md"),
)

# Directories under docs/ that are historical records, not live documentation.
DOCS_EXCLUDED_DIRS = ("superpowers",)

_FENCE = re.compile(r"^\s*(```|~~~)")
# Inline code spans hold link-shaped *examples* (CLAUDE.md documents the
# `![alt](diagrams/foo.svg)` form), so they are not links.
_CODE_SPAN = re.compile(r"`[^`]*`")
# Markdown inline links and images: ](target) — the target may carry a title.
_LINK = re.compile(r"!?\]\(\s*<?([^)>\s]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")
# Link reference definitions: `[label]: target "title"`. A reference-style link
# resolves through one of these, so the definition's target is the real link.
_LINK_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?([^>\s]+)>?(?:\s+[\"'(][^\"')]*[\"')])?\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# An explicit HTML anchor pins a slug against a heading rewrite; docs/ReSTIR.md
# already uses this form.
_HTML_ANCHOR = re.compile(r"<a\s+(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
# GitHub drops everything that is not alphanumeric, space, hyphen or underscore.
_SLUG_DROP = re.compile(r"[^\w\s-]", re.UNICODE)


def _live_docs() -> list[str]:
    """Every live Markdown document, as a repo-relative path."""
    found = [p for p in ROOT_DOCS if os.path.exists(os.path.join(REPO, p))]
    docs_root = os.path.join(REPO, "docs")
    for dirpath, dirnames, filenames in os.walk(docs_root):
        dirnames[:] = [d for d in dirnames if d not in DOCS_EXCLUDED_DIRS]
        for name in sorted(filenames):
            if name.endswith(".md"):
                full = os.path.join(dirpath, name)
                found.append(os.path.relpath(full, REPO))
    return sorted(found)


def _strip_fences(text: str) -> list[str]:
    """Blank out fenced code blocks so a code sample cannot look like a link."""
    lines = text.splitlines()
    out, in_fence = [], False
    for line in lines:
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append("")
            continue
        out.append("" if in_fence else line)
    return out


def _slug(heading_text: str) -> str:
    """GitHub's anchor slug: lowercase, drop punctuation, spaces to hyphens."""
    text = heading_text.strip().lower()
    # Markdown emphasis and code markers are punctuation to GitHub's slugger.
    text = _SLUG_DROP.sub("", text)
    return text.replace(" ", "-")


def _anchors(path: str) -> set[str]:
    """Every anchor a Markdown file exposes, including GitHub's -1/-2 suffixes."""
    with open(path, encoding="utf-8") as fh:
        lines = _strip_fences(fh.read())
    seen: Counter[str] = Counter()
    anchors: set[str] = set()
    for line in lines:
        anchors.update(_HTML_ANCHOR.findall(line))
        match = _HEADING.match(line)
        if not match:
            continue
        base = _slug(match.group(2))
        if not base:
            continue
        count = seen[base]
        seen[base] += 1
        anchors.add(base if count == 0 else f"{base}-{count}")
    return anchors


def _links(rel_doc: str) -> list[str]:
    with open(os.path.join(REPO, rel_doc), encoding="utf-8") as fh:
        lines = _strip_fences(fh.read())
    out = []
    for line in lines:
        stripped = _CODE_SPAN.sub("", line)
        out.extend(m.group(1) for m in _LINK.finditer(stripped))
        definition = _LINK_DEF.match(stripped)
        if definition:
            out.append(definition.group(1))
    return out


def _is_relative(target: str) -> bool:
    lowered = target.lower()
    if lowered.startswith(("http://", "https://", "mailto:", "ftp://", "//")):
        return False
    return not target.startswith("/")


LIVE_DOCS = _live_docs()


def test_live_doc_set_is_not_empty():
    """A silently empty walk would make every check below vacuous."""
    assert len(LIVE_DOCS) >= 15, LIVE_DOCS
    assert "README.md" in LIVE_DOCS
    assert os.path.join("docs", "Architecture.md") in LIVE_DOCS


def test_index_lists_every_reference_document():
    """docs/README.md must link every reference document in docs/.

    Scope is the top level of `docs/`. Nested directories hold generated
    artifacts — `docs/diagrams/` carries the SVG generators and their result
    reports, `docs/superpowers/` records history — and neither is a reference
    document the index should enumerate. Their links are still checked above.
    """
    index = os.path.join(REPO, "docs", "README.md")
    linked = {
        os.path.basename(t.split("#", 1)[0])
        for t in _links(os.path.join("docs", "README.md"))
        if _is_relative(t)
    }
    on_disk = {
        n for n in os.listdir(os.path.join(REPO, "docs"))
        if n.endswith(".md") and n != "README.md"
    }
    assert on_disk, "no reference documents found — the check would be vacuous"
    missing = sorted(on_disk - linked)
    assert not missing, f"{index} does not link: {missing}"


def test_negative_control(tmp_path):
    """The checks must actually fail on a broken link — not pass vacuously."""
    target = tmp_path / "target.md"
    target.write_text(
        "## Real Heading (`mod.py`)\n"
        '<a id="pinned"></a>\n'
        "## Real Heading (`mod.py`)\n"
        "```\n](fenced/not-a-link.md)\n```\n",
        encoding="utf-8",
    )
    anchors = _anchors(str(target))
    assert "real-heading-modpy" in anchors  # backticks and dots dropped
    assert "real-heading-modpy-1" in anchors  # duplicate gets GitHub's suffix
    assert "pinned" in anchors  # explicit HTML anchor honoured
    assert "missing-heading" not in anchors

    source = tmp_path / "source.md"
    source.write_text(
        "[ok](target.md#pinned) [gone](nope.md) "
        "[stale](target.md#missing-heading) `[example](docs/foo.svg)`\n"
        "A reference-style link to [the guide][g].\n"
        "\n"
        "[g]: also-nope.md\n",
        encoding="utf-8",
    )
    links = []
    for line in _strip_fences(source.read_text(encoding="utf-8")):
        stripped = _CODE_SPAN.sub("", line)
        links.extend(m.group(1) for m in _LINK.finditer(stripped))
        d = _LINK_DEF.match(stripped)
        if d:
            links.append(d.group(1))
    assert links == [
        "target.md#pinned",
        "nope.md",
        "target.md#missing-heading",
        "also-nope.md",  # a reference definition IS a link
    ]
    assert not (tmp_path / "nope.md").exists()
    assert "missing-heading" not in anchors
    # A fenced code block never contributes a link.
    assert "fenced/not-a-link.md" not in [
        m.group(1)
        for line in _strip_fences(target.read_text(encoding="utf-8"))
        for m in _LINK.finditer(line)
    ]


@pytest.mark.parametrize("doc", LIVE_DOCS)
def test_relative_links_resolve(doc):
    """Every relative link target names a file that exists."""
    base = os.path.dirname(os.path.join(REPO, doc))
    broken = []
    for target in _links(doc):
        if not _is_relative(target):
            continue
        path = target.split("#", 1)[0]
        if not path:  # a bare #anchor points at this document
            continue
        if not os.path.exists(os.path.normpath(os.path.join(base, path))):
            broken.append(target)
    assert not broken, f"{doc}: link target does not exist: {broken}"


@pytest.mark.parametrize("doc", LIVE_DOCS)
def test_link_anchors_resolve(doc):
    """Every #anchor on a relative link matches a heading in the target file."""
    base = os.path.dirname(os.path.join(REPO, doc))
    broken = []
    for target in _links(doc):
        if not _is_relative(target) or "#" not in target:
            continue
        path, _, anchor = target.partition("#")
        if not anchor:
            continue
        full = os.path.normpath(os.path.join(base, path)) if path else os.path.join(REPO, doc)
        if not full.endswith(".md") or not os.path.exists(full):
            continue  # a missing file is the other test's failure
        if anchor not in _anchors(full):
            broken.append(target)
    assert not broken, f"{doc}: anchor not found in target: {broken}"
