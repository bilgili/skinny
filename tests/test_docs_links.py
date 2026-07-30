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

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
# Inline code spans hold link-shaped *examples* (CLAUDE.md documents the
# `![alt](diagrams/foo.svg)` form), so they are not links.
_CODE_SPAN = re.compile(r"`[^`]*`")
# Markdown inline links and images: ](target) — the target may carry a title.
# CommonMark allows an angle-bracket destination, which is the only form that
# may contain spaces: `[x](<My Guide.md>)`. Match it first, or the bare-word
# alternative silently yields nothing and the link goes unchecked.
#
# A bare destination may also carry BALANCED parentheses — `[x](plan_(draft).md)`
# is valid — so the third alternative allows one level of nesting. Stopping at
# the first `)` would extract `plan_(draft` and report a missing file for a
# document that exists.
_LINK = re.compile(
    r"!?\]\(\s*(?:<([^<>\n]*)>|((?:[^()<>\s]|\([^()<>\s]*\))+))"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)
# Link reference definitions: `[label]: target "title"`. A reference-style link
# resolves through one of these, so the definition's target is the real link.
_LINK_DEF = re.compile(
    r"^ {0,3}\[[^\]]+\]:\s*(?:<([^<>\n]*)>|([^<>\s]+))(?:\s+[\"'(][^\"')]*[\"')])?\s*$"
)
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
    """Blank out fenced code blocks so a code sample cannot look like a link.

    CommonMark closes a fence only with the SAME delimiter character, repeated
    at least as many times as the opener. A four-backtick fence showing a
    three-backtick Markdown example must therefore stay open across the inner
    ``` line — toggling on every fence-looking line would treat the sample text
    as real links.
    """
    out = []
    opener = None  # (char, length) while inside a fence
    for line in text.splitlines():
        match = _FENCE.match(line)
        if match:
            char, run = match.group(1)[0], len(match.group(1))
            if opener is None:
                opener = (char, run)
                out.append("")
                continue
            if char == opener[0] and run >= opener[1]:
                opener = None
                out.append("")
                continue
        out.append("" if opener else line)
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
    # github-slugger's disambiguation: on a collision, append `-N` where N is
    # the occurrence count of the ORIGINAL slug, then retry against every slug
    # already emitted. Headings `Foo`, `Foo`, `Foo-1` give foo, foo-1, foo-1-1
    # — a per-base counter would wrongly give foo-1 twice.
    occurrences: Counter[str] = Counter()
    emitted: set[str] = set()
    anchors: set[str] = set()
    for line in lines:
        anchors.update(_HTML_ANCHOR.findall(line))
        match = _HEADING.match(line)
        if not match:
            continue
        base = _slug(match.group(2))
        if not base:
            continue
        result = base
        while result in emitted:
            occurrences[base] += 1
            result = f"{base}-{occurrences[base]}"
        emitted.add(result)
        anchors.add(result)
    return anchors


def _links(rel_doc: str) -> list[str]:
    with open(os.path.join(REPO, rel_doc), encoding="utf-8") as fh:
        lines = _strip_fences(fh.read())
    out = []
    for line in lines:
        stripped = _CODE_SPAN.sub("", line)
        # Each pattern has two destination groups: <angle-bracket> or bare.
        out.extend(m.group(1) or m.group(2) for m in _LINK.finditer(stripped))
        definition = _LINK_DEF.match(stripped)
        if definition:
            out.append(definition.group(1) or definition.group(2))
    return [t for t in out if t]


def _exists_case_exact(path: str) -> bool:
    """os.path.exists, but honouring case even on a case-insensitive volume.

    This macOS checkout resolves `docs/architecture.md` to `docs/Architecture.md`,
    so a mis-cased link passes here and 404s on GitHub and on Linux. Walk the
    path from the repo root and require each component to appear in its parent's
    listing exactly as written.
    """
    if not os.path.exists(path):
        return False
    current = os.path.abspath(REPO)
    rest = os.path.relpath(os.path.abspath(path), current)
    if rest.startswith(os.pardir):
        return True  # outside the repo — existence is all we can claim
    for part in rest.split(os.sep):
        if part in (os.curdir, ""):
            continue
        try:
            if part not in os.listdir(current):
                return False
        except OSError:
            return False
        current = os.path.join(current, part)
    return True


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


INDEX_HEADING = "## Documentation"


def _readme_split() -> tuple[list[str], list[str]]:
    """README.md's `## Documentation` lines, and every line outside it.

    Returns two line lists rather than joined strings on purpose: `_strip_fences`
    drops line terminators, so re-joining and then string-replacing one part out
    of the other silently matches nothing.
    """
    lines = _strip_fences(open(os.path.join(REPO, "README.md"), encoding="utf-8").read())
    starts = [i for i, ln in enumerate(lines) if ln.startswith(INDEX_HEADING)]
    assert len(starts) == 1, f"README.md must have exactly one {INDEX_HEADING!r}"
    start = starts[0]
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    return lines[start:end], lines[:start] + lines[end:]


def _index_section() -> list[str]:
    return _readme_split()[0]


def test_index_lists_every_reference_document():
    """README.md must link every reference document in docs/.

    `README.md` IS the index (change `readme-as-docs-index`): a reader arrives
    there, so it must say where everything lives without a redirect.

    Scope is the top level of `docs/`. Nested directories hold generated
    artifacts — `docs/diagrams/` carries the SVG generators and their result
    reports, `docs/superpowers/` records history — and neither is a reference
    document the index should enumerate. Their links are still checked above.
    """
    docs_dir = os.path.join(REPO, "docs")
    # Only the Documentation section counts. README also links documents from its
    # intro, features, and quick start, and those must not stand in for an index
    # row — otherwise deleting a row still passes and the check is decorative.
    section = _index_section()
    # Targets are relative to the repo root, since the index is README.md.
    # Resolve each one, so a link to a nested report that happens to share a
    # filename cannot stand in for the top-level document.
    linked = set()
    for line in section:
        for m in _LINK.finditer(_CODE_SPAN.sub("", line)):
            target = m.group(1) or m.group(2)
            if not target or not _is_relative(target):
                continue
            resolved = os.path.normpath(os.path.join(REPO, target.split("#", 1)[0]))
            if os.path.dirname(resolved) == docs_dir:
                linked.add(os.path.basename(resolved))
    on_disk = {n for n in os.listdir(docs_dir) if n.endswith(".md")}
    assert on_disk, "no reference documents found — the check would be vacuous"
    missing = sorted(on_disk - linked)
    assert not missing, f"README.md § Documentation does not link: {missing}"


def test_index_check_ignores_links_outside_the_index_section():
    """A link in the intro or quick start must not count as an index row.

    README links some documents twice — once in prose, once in the index. If the
    completeness check scanned the whole file, deleting an index row would still
    pass because the prose link covers it, and the check would be decorative.
    """
    section, outside = _readme_split()
    assert section and outside, "README.md split produced an empty half"

    def linked(lines):
        found = set()
        for line in lines:
            for m in _LINK.finditer(_CODE_SPAN.sub("", line)):
                target = m.group(1) or m.group(2)
                if target and _is_relative(target):
                    found.add(os.path.basename(target.split("#", 1)[0]))
        return found

    inside_names, outside_names = linked(section), linked(outside)
    # The scoping only matters if some document IS linked twice. Assert that,
    # or this guard would quietly stop testing anything.
    both = sorted(inside_names & outside_names)
    assert both, (
        "no document is linked both inside and outside the index — "
        "the section scoping in test_index_lists_every_reference_document "
        "is no longer load-bearing, so either it or this guard is wrong"
    )
    # The halves must be real, disjoint slices — not the whole file twice, which
    # is what silently happened when this guard joined the lines and used
    # str.replace (fences drop line terminators, so nothing matched).
    assert any(ln.startswith(INDEX_HEADING) for ln in section)
    assert not any(ln.startswith(INDEX_HEADING) for ln in outside)
    # Every index row names a doc; the outside half must not carry them all.
    assert len(inside_names) > len(outside_names)


def test_there_is_exactly_one_index():
    """docs/README.md must not come back — the index has one home.

    Two files claiming to be the index is the shape this change removed; the
    second one is a redirect the reader has to follow.
    """
    stray = os.path.join(REPO, "docs", "README.md")
    assert not os.path.exists(stray), (
        "docs/README.md exists again — README.md is the index; "
        "add new documents to its Documentation section instead"
    )


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

    # A heading whose own slug collides with a generated suffix: github-slugger
    # retries against every emitted slug, so this is foo, foo-1, foo-1-1 — NOT
    # foo-1 twice, which is what a naive per-base counter produces.
    collide = tmp_path / "collide.md"
    collide.write_text("# Foo\n# Foo\n# Foo-1\n", encoding="utf-8")
    assert _anchors(str(collide)) == {"foo", "foo-1", "foo-1-1"}

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
        links.extend(m.group(1) or m.group(2) for m in _LINK.finditer(stripped))
        d = _LINK_DEF.match(stripped)
        if d:
            links.append(d.group(1) or d.group(2))
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
        m.group(1) or m.group(2)
        for line in _strip_fences(target.read_text(encoding="utf-8"))
        for m in _LINK.finditer(line)
    ]

    # A four-backtick fence stays open across an inner ``` line (CommonMark
    # closes only on the same character, at the opener's length or longer), so
    # the sample link inside it is not a link.
    nested = tmp_path / "nested.md"
    nested.write_text(
        "````markdown\n"
        "```\n"
        "[sample](inner/not-a-link.md)\n"
        "```\n"
        "````\n"
        "[real](target.md)\n",
        encoding="utf-8",
    )
    kept = [
        m.group(1) or m.group(2)
        for line in _strip_fences(nested.read_text(encoding="utf-8"))
        for m in _LINK.finditer(line)
    ]
    assert kept == ["target.md"], kept

    # An angle-bracket destination is the only inline form that may hold a
    # space. A pattern that rejects it yields NO target, so the link goes
    # unchecked and a missing file passes.
    angled = tmp_path / "angled.md"
    angled.write_text(
        "[guide](<My Guide.md>) and [plain](<tight.md>)\n\n[r]: <Ref Doc.md>\n",
        encoding="utf-8",
    )
    got = []
    for line in _strip_fences(angled.read_text(encoding="utf-8")):
        got.extend(m.group(1) or m.group(2) for m in _LINK.finditer(line))
        d = _LINK_DEF.match(line)
        if d:
            got.append(d.group(1) or d.group(2))
    assert got == ["My Guide.md", "tight.md", "Ref Doc.md"], got

    # A bare destination may carry balanced parentheses. Stopping at the first
    # `)` would extract `plan_(draft` and fail on valid Markdown.
    parens = tmp_path / "parens.md"
    parens.write_text("[design](plan_(draft).md) then [plain](x.md)\n", encoding="utf-8")
    got = [
        m.group(1) or m.group(2)
        for line in _strip_fences(parens.read_text(encoding="utf-8"))
        for m in _LINK.finditer(line)
    ]
    assert got == ["plan_(draft).md", "x.md"], got

    # Case matters on GitHub and on Linux even when this volume forgives it.
    # Checked inside the repo, where the walk applies — tmp_path is outside it.
    right = os.path.join(REPO, "docs", "Architecture.md")
    wrong = os.path.join(REPO, "docs", "architecture.md")
    assert _exists_case_exact(right)
    assert not _exists_case_exact(wrong)
    if not os.path.exists(wrong):
        pytest.skip("case-sensitive filesystem: os.path.exists already rejects it")


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
        if not _exists_case_exact(os.path.normpath(os.path.join(base, path))):
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
        if not full.endswith(".md") or not _exists_case_exact(full):
            continue  # a missing file is the other test's failure
        if anchor not in _anchors(full):
            broken.append(target)
    assert not broken, f"{doc}: anchor not found in target: {broken}"
