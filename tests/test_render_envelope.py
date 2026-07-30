"""Render-envelope predicate: the shared "does combo X run" statement.

Group 1 of change ``unified-render-envelope-predicate`` is the *behaviour-
preservation gate* (design D7): before any production code moves, the full
cartesian query space is evaluated through the **current** enforcement layers
and committed as a JSON fixture. Every later group re-runs
:func:`test_combo_is_valid_matches_snapshot` (and the CLI/spectral-envelope
siblings) against that fixture, so a refactor that silently shifts the envelope
fails here.

Two maps, because no single pre-port function sees every axis:

``combo_is_valid``
    ``parity.combo_is_valid`` over integrator × execution mode × proposal
    subsets × reuse × spectral × material class × megakernel budget. It has no
    ``online_training`` axis, so that axis is not replicated here (it would
    duplicate every row); it is covered by the CLI map below.
``cli_guards``
    the four ``cli_common`` refusal guards over the flag-visible axes —
    integrator × execution mode × proposals × reuse × spectral ×
    online_training — recording the exact ``SystemExit`` text (or ``None`` for
    "accepted"). This is where ``online_training`` and the per-guard prose are
    pinned.

Both maps are additionally recorded under the capability-flag-off variants
(``SPECTRAL_IMPLEMENTED=False``, ``MLT_IMPLEMENTED=False``), and
``parity.all_combos()`` — the rendered-set enumerator, deliberately sparser
than the query space — is snapshotted as a label list.

Regenerate (only when an envelope change is *intended*)::

    PYTHONPATH=src .venv/bin/python tests/test_render_envelope.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "assets", "render_envelope_snapshot.json")

# ─── the full cartesian query space ────────────────────────────────────────

INTEGRATORS = ("path", "bdpt", "sppm", "mlt")
EXECUTION_MODES = ("megakernel", "wavefront")
PROPOSAL_SUBSETS = ((), ("env",), ("neural",), ("env", "neural"))
REUSE_VALUES = ("none", "restir-di")
MATERIAL_CLASSES = ("flat", "subsurface", "skin", "volume")


def _combo_points():
    """(integrator, mode, proposals, reuse, spectral, material_class, megakernel_ok)."""
    return itertools.product(
        INTEGRATORS, EXECUTION_MODES, PROPOSAL_SUBSETS, REUSE_VALUES,
        (False, True), MATERIAL_CLASSES, (True, False),
    )


def _cli_points():
    """(integrator, mode, proposals, reuse, spectral, online_training)."""
    return itertools.product(
        INTEGRATORS, EXECUTION_MODES, PROPOSAL_SUBSETS, REUSE_VALUES,
        (False, True), (False, True),
    )


def _proposals_flag(proposals: tuple[str, ...]) -> str:
    """The ``--proposals`` string for a proposal subset (``bsdf`` always present)."""
    return ",".join(("bsdf", *proposals))


def _key(*parts) -> str:
    return "|".join(
        "+".join(p) if isinstance(p, tuple) else str(p) for p in parts
    )


# ─── evaluation of the CURRENT layers (the snapshot source) ────────────────

def _eval_combo_map() -> dict[str, list]:
    from skinny.pbrt.parity import RenderCombo, SceneSpec, combo_is_valid, spectral_envelope

    out: dict[str, list] = {}
    for integ, mode, props, reuse, spectral, mat, mega_ok in _combo_points():
        combo = RenderCombo(integ, mode, props, reuse, spectral=spectral)
        scene = SceneSpec(
            name="q", file="", ref="", width=8, height=8, spp=1,
            relmse_tol=1.0, flip_tol=1.0,
            material_class=mat, megakernel_ok=mega_ok,
        )
        ok, reason = combo_is_valid(combo, scene)
        senv_ok, senv_reason = spectral_envelope(combo, scene)
        out[_key(integ, mode, props, reuse, spectral, mat, mega_ok)] = [
            bool(ok), reason, bool(senv_ok), senv_reason,
        ]
    return out


def _eval_cli_map() -> dict[str, dict]:
    from skinny.cli_common import (
        reject_mlt_unsupported,
        reject_spectral_unsupported,
        reject_sppm_without_wavefront,
        validate_render_flags,
    )

    def _run(fn, *a, **kw):
        try:
            fn(*a, **kw)
        except SystemExit as exc:
            return str(exc)
        return None

    out: dict[str, dict] = {}
    for integ, mode, props, reuse, spectral, online in _cli_points():
        flag = _proposals_flag(props)
        ns = argparse.Namespace(
            integrator=integ, execution_mode=mode, proposals=flag,
            reuse=reuse, spectral=spectral, online_training=online,
            width=64, height=64,
        )
        out[_key(integ, mode, props, reuse, spectral, online)] = {
            "validate_render_flags": _run(validate_render_flags, ns),
            "reject_sppm_without_wavefront": _run(
                reject_sppm_without_wavefront, integ, mode),
            "reject_mlt_unsupported": _run(
                reject_mlt_unsupported, integ, mode, spectral=spectral,
                proposals=flag, reuse=reuse, online_training=online),
            "reject_spectral_unsupported": _run(
                reject_spectral_unsupported, spectral, integ, mode, flag, reuse),
        }
    return out


def _eval_all(monkeypatch=None) -> dict:
    from skinny.pbrt import parity

    return {
        "combo_is_valid": _eval_combo_map(),
        "cli_guards": _eval_cli_map(),
        "all_combos": [c.label for c in parity.all_combos()],
    }


def _capability_variants():
    """(fixture section name, {module attr → value}) for the flag-off snapshots."""
    return (
        ("flags_on", {}),
        ("spectral_off", {"spectral_capability": False}),
        ("mlt_off", {"mlt_capability": False}),
    )


def _with_flags(patch: dict, fn):
    """Run *fn* with the named capability flags forced off, restoring after."""
    from skinny import mlt_capability, spectral_capability

    mods = {"spectral_capability": (spectral_capability, "SPECTRAL_IMPLEMENTED"),
            "mlt_capability": (mlt_capability, "MLT_IMPLEMENTED")}
    saved = {}
    try:
        for name, value in patch.items():
            mod, attr = mods[name]
            saved[name] = getattr(mod, attr)
            setattr(mod, attr, value)
        return fn()
    finally:
        for name, value in saved.items():
            mod, attr = mods[name]
            setattr(mod, attr, value)


def build_snapshot() -> dict:
    return {
        section: _with_flags(patch, _eval_all)
        for section, patch in _capability_variants()
    }


# Reasons and refusal prose repeat across thousands of query points (the CLI
# messages are multi-sentence), so the fixture stores string *indices* into a
# shared table (~1 MB → ~90 kB) and the loader re-expands them. Purely a storage
# concern: the compared value is always the full string.

def _intern(snapshot: dict) -> dict:
    table: list[str] = []
    index: dict[str, int] = {}

    def ref(msg):
        if msg is None:
            return None
        if msg not in index:
            index[msg] = len(table)
            table.append(msg)
        return index[msg]

    packed = {
        section: {
            **data,
            "combo_is_valid": {k: [ok, ref(r), senv_ok, ref(senv_r)]
                               for k, (ok, r, senv_ok, senv_r)
                               in data["combo_is_valid"].items()},
            "cli_guards": {k: {g: ref(m) for g, m in v.items()}
                           for k, v in data["cli_guards"].items()},
        }
        for section, data in snapshot.items()
    }
    packed["_messages"] = table
    return packed


def _expand(packed: dict) -> dict:
    table = packed["_messages"]
    deref = lambda i: None if i is None else table[i]  # noqa: E731
    return {
        section: {
            **data,
            "combo_is_valid": {k: [ok, deref(r), senv_ok, deref(senv_r)]
                               for k, (ok, r, senv_ok, senv_r)
                               in data["combo_is_valid"].items()},
            "cli_guards": {k: {g: deref(i) for g, i in v.items()}
                           for k, v in data["cli_guards"].items()},
        }
        for section, data in packed.items() if section != "_messages"
    }


# ─── the equivalence gate ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def snapshot() -> dict:
    if not os.path.exists(FIXTURE):
        pytest.fail(f"missing envelope snapshot {FIXTURE} — regenerate with "
                    f"`PYTHONPATH=src python {__file__}`")
    with open(FIXTURE, encoding="utf-8") as fh:
        return _expand(json.load(fh))


def _diff(expected: dict, actual: dict, limit: int = 8) -> str:
    keys = sorted(set(expected) | set(actual))
    bad = [k for k in keys if expected.get(k) != actual.get(k)]
    head = "\n".join(f"  {k}\n    was {expected.get(k)!r}\n    now {actual.get(k)!r}"
                     for k in bad[:limit])
    return f"{len(bad)} divergent key(s):\n{head}"


@pytest.mark.parametrize("section,patch", _capability_variants())
def test_combo_is_valid_matches_snapshot(snapshot, section, patch):
    """Every point of the query space keeps its (valid, reason) — and the
    spectral envelope keeps its own reason precedence."""
    actual = _with_flags(patch, _eval_combo_map)
    expected = snapshot[section]["combo_is_valid"]
    assert actual == expected, _diff(expected, actual)


@pytest.mark.parametrize("section,patch", _capability_variants())
def test_cli_guards_match_snapshot(snapshot, section, patch):
    """Every CLI guard keeps its accept/refuse decision AND byte-identical prose."""
    actual = _with_flags(patch, _eval_cli_map)
    expected = snapshot[section]["cli_guards"]
    assert actual == expected, _diff(expected, actual)


@pytest.mark.parametrize("section,patch", _capability_variants())
def test_all_combos_rendered_set_unchanged(snapshot, section, patch):
    """`all_combos()` keeps its rendered-set enumeration role, unchanged."""
    from skinny.pbrt import parity

    actual = _with_flags(patch, lambda: [c.label for c in parity.all_combos()])
    assert actual == snapshot[section]["all_combos"]


# ─── the predicate itself ──────────────────────────────────────────────────

from skinny import render_envelope as re  # noqa: E402
from skinny.render_envelope import EnvelopeQuery as Q  # noqa: E402
from skinny.render_envelope import evaluate  # noqa: E402

#: One accepting and one refusing query per reason code. The accepting query is
#: the refusing one with the single offending axis removed, so each row isolates
#: exactly the rule it names.
CODE_CASES = [
    (re.UNKNOWN_INTEGRATOR, Q(integrator="vcm"), Q(integrator="path")),
    (re.UNKNOWN_EXECUTION_MODE, Q(execution_mode="auto"), Q(execution_mode="wavefront")),
    (re.SPPM_WAVEFRONT_ONLY, Q("sppm", "megakernel"), Q("sppm", "wavefront")),
    (re.MLT_WAVEFRONT_ONLY, Q("mlt", "megakernel"), Q("mlt", "wavefront")),
    (re.MLT_NO_PROPOSALS, Q("mlt", "wavefront", ("env",)), Q("mlt", "wavefront")),
    (re.MLT_NO_REUSE, Q("mlt", "wavefront", (), "restir-di"), Q("mlt", "wavefront")),
    (re.MLT_FLAT_ONLY, Q("mlt", "wavefront", material_class="skin"), Q("mlt", "wavefront")),
    (re.NEURAL_WAVEFRONT_ONLY, Q("path", "megakernel", ("neural",)),
     Q("path", "wavefront", ("neural",))),
    (re.NEURAL_PATH_ONLY, Q("bdpt", "wavefront", ("neural",)),
     Q("path", "wavefront", ("neural",))),
    (re.NEURAL_FLAT_ONLY, Q("path", "wavefront", ("neural",), material_class="skin"),
     Q("path", "wavefront", ("neural",))),
    (re.ENV_PATH_ONLY, Q("bdpt", "wavefront", ("env",)), Q("path", "wavefront", ("env",))),
    (re.ENV_FLAT_ONLY, Q("path", "wavefront", ("env",), material_class="skin"),
     Q("path", "wavefront", ("env",))),
    (re.REUSE_WAVEFRONT_ONLY, Q("path", "megakernel", (), "restir-di"),
     Q("path", "wavefront", (), "restir-di")),
    (re.REUSE_PATH_ONLY, Q("bdpt", "wavefront", (), "restir-di"),
     Q("path", "wavefront", (), "restir-di")),
    (re.VOLUME_INTEGRATOR_PATH_ONLY, Q("bdpt", "wavefront", material_class="volume"),
     Q("path", "wavefront", material_class="volume")),
    (re.VOLUME_NO_REUSE, Q("path", "wavefront", (), "restir-di", material_class="volume"),
     Q("path", "wavefront", material_class="volume")),
    (re.ONLINE_TRAINING_PATH_ONLY, Q("bdpt", "wavefront", online_training=True),
     Q("path", "wavefront", online_training=True)),
    (re.SPECTRAL_INTEGRATOR_UNSUPPORTED, Q("vcm", "wavefront", spectral=True),
     Q("path", "wavefront", spectral=True)),
    (re.SPECTRAL_NO_NEURAL, Q("path", "wavefront", ("neural",), spectral=True),
     Q("path", "wavefront", ("env",), spectral=True)),
    (re.SPECTRAL_ENV_PATH_ONLY, Q("bdpt", "wavefront", ("env",), spectral=True),
     Q("path", "wavefront", ("env",), spectral=True)),
    (re.SPECTRAL_NO_REUSE, Q("path", "wavefront", (), "restir-di", spectral=True),
     Q("path", "wavefront", spectral=True)),
    (re.SPECTRAL_FLAT_ONLY, Q("path", "wavefront", spectral=True, material_class="skin"),
     Q("path", "wavefront", spectral=True)),
    (re.MEGAKERNEL_BUDGET, Q("path", "megakernel", megakernel_ok=False),
     Q("path", "wavefront", megakernel_ok=False)),
]


@pytest.mark.parametrize("code,refused,accepted", CODE_CASES,
                         ids=[c[0] for c in CODE_CASES])
def test_each_code_has_a_refusing_and_an_accepting_query(code, refused, accepted):
    assert code in evaluate(refused).codes
    assert code not in evaluate(accepted).codes


def test_capability_flags_are_read_live():
    """The two "not yet wired" codes appear only while their flag is off, with no
    consumer-side special-casing — the flags are read at evaluation time."""
    mlt, spec = Q("mlt", "wavefront"), Q("path", "wavefront", spectral=True)
    assert evaluate(mlt).ok and evaluate(spec).ok
    assert re.MLT_NOT_WIRED in _with_flags({"mlt_capability": False},
                                           lambda: evaluate(mlt)).codes
    assert re.SPECTRAL_NOT_WIRED in _with_flags({"spectral_capability": False},
                                                lambda: evaluate(spec)).codes
    # …and the flags are restored, not latched.
    assert evaluate(mlt).ok and evaluate(spec).ok


def test_verdict_lists_every_violation_not_only_the_first():
    """The D3 counterexample: `bdpt` + neural under the megakernel violates both
    the neural-wavefront-only rule (which parity reports) and the path-only rule
    (whose prose the CLI must print) — and the CLI *accepts* `path` + neural under
    the megakernel, which violates only the first."""
    verdict = evaluate(Q("bdpt", "megakernel", ("neural",)))
    assert verdict.codes == (re.NEURAL_WAVEFRONT_ONLY, re.NEURAL_PATH_ONLY)
    # Canonical order drives parity; the CLI selects the code it owns instead.
    assert verdict.first()[1] == "neural proposal is wavefront-only"
    assert verdict.first_code(re.NEURAL_PATH_ONLY) == re.NEURAL_PATH_ONLY
    assert evaluate(Q("path", "megakernel", ("neural",))).first_code(
        *re.CLI_GUARD_CODES["validate_render_flags"]) is None


def test_mlt_guard_precedence_is_opposite_to_parity():
    """`reject_mlt_unsupported` reports not-yet-wired before the megakernel
    refusal; the matrix reports them the other way round. Same verdict, two
    consumer-owned precedences."""
    verdict = _with_flags({"mlt_capability": False},
                          lambda: evaluate(Q("mlt", "megakernel")))
    assert verdict.first()[0] is False
    assert verdict.first()[1] == "MLT is wavefront-only"
    assert verdict.first_code(*re.CLI_GUARD_CODES["reject_mlt_unsupported"]) == \
        re.MLT_NOT_WIRED


def test_code_ownership_table_is_complete_and_disjoint():
    """Every declared code is owned by exactly one place — a CLI guard, the
    renderer scene gate, or the recorded unowned set — and every owned code
    exists."""
    declared = set(re.ALL_CODES)
    assert len(re.ALL_CODES) == len(declared), "duplicate entry in ALL_CODES"
    owned = [c for codes in re.CLI_GUARD_CODES.values() for c in codes]
    # ONLINE_TRAINING_PATH_ONLY is deliberately shared: the mlt and bdpt guards
    # each print their own prose for it.
    assert sorted(set(owned)) == sorted(
        {c for c in owned if c != re.ONLINE_TRAINING_PATH_ONLY}
        | {re.ONLINE_TRAINING_PATH_ONLY})
    assert set(owned) <= declared
    assert set(re.RENDERER_SCENE_CODES) <= declared
    assert set(re.CLI_UNOWNED_CODES) <= declared
    assert set(owned) | set(re.CLI_UNOWNED_CODES) == declared
    assert not set(owned) & set(re.CLI_UNOWNED_CODES)
    # The renderer's code is CLI-unowned by construction (the CLI cannot see it).
    assert set(re.RENDERER_SCENE_CODES) <= set(re.CLI_UNOWNED_CODES)


def test_module_imports_without_the_renderer_parity_or_cli():
    """The predicate is a leaf above the two capability modules — importing it
    must not drag in the renderer, the parity harness, or the CLI layer (that is
    what lets all three import it back)."""
    import subprocess

    src = os.path.join(os.path.dirname(__file__), "..", "src")
    code = (
        "import sys; import skinny.render_envelope;"
        "loaded = [m for m in sys.modules if m.startswith('skinny.')];"
        "assert 'skinny.renderer' not in loaded, loaded;"
        "assert 'skinny.cli_common' not in loaded, loaded;"
        "assert 'skinny.pbrt.parity' not in loaded, loaded;"
        "print('ok')"
    )
    env = dict(os.environ, PYTHONPATH=src)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# ─── the renderer's spectral scene gate (D4) ───────────────────────────────

@pytest.mark.parametrize("material_class", ("subsurface", "skin", "volume"))
def test_spectral_scene_gate_refuses_each_non_flat_class(material_class):
    verdict = evaluate(Q("path", "wavefront", spectral=True,
                         material_class=material_class))
    assert verdict.first_code(*re.RENDERER_SCENE_CODES) == re.SPECTRAL_FLAT_ONLY


@pytest.mark.parametrize("material_class", ("subsurface", "skin", "volume"))
@pytest.mark.parametrize("integrator", ("path", "bdpt", "mlt"))
def test_scene_gate_scans_only_its_own_code(integrator, material_class):
    """A non-spectral scene must trip NO renderer-gate refusal, however many other
    rules it violates — the gate scans only the flat-material code, so MLT or
    neural on non-flat materials stays a parity skip / runtime fallback."""
    verdict = evaluate(Q(integrator, "wavefront", ("neural",),
                         material_class=material_class))
    assert not verdict.ok  # it does violate other rules …
    assert verdict.first_code(*re.RENDERER_SCENE_CODES) is None  # … but not this one


def test_renderer_material_type_mapping_refuses_every_non_flat_code():
    """The pinned int-code → material-class mapping keeps every non-flat packed
    type refused under --spectral, including debug=2 / python=3, which are not
    parity material classes — and leaves flat=1 alone."""
    for type_code in (0, 2, 3, 4, 5):
        assert re.spectral_refuses_material_types([type_code]), type_code
    assert not re.spectral_refuses_material_types([1])
    assert not re.spectral_refuses_material_types([])
    # An unmapped code still refuses (the fallback is a non-flat class).
    assert re.spectral_refuses_material_types([99])
    assert set(re.MATERIAL_CLASS_FOR_TYPE_CODE) == {0, 1, 2, 3, 4, 5}


# ─── doc-sync (D8) ─────────────────────────────────────────────────────────
#
# The CLAUDE.md / RenderingModes.md compatibility tables document the predicate
# rather than being mirrored into it. This check is deliberately shallow — it
# catches "someone changed the envelope and forgot the docs", not prose
# equivalence.
#
# The tuple names the documents that HOLD the tables. The README table moved to
# docs/RenderingModes.md in change `docs-split-large-docs`; moving a table again
# means updating this tuple in the same change.

_REPO = os.path.join(os.path.dirname(__file__), "..")
DOC_TABLES = (os.path.join("docs", "RenderingModes.md"), "CLAUDE.md")


def _doc_text(name: str) -> str:
    with open(os.path.join(_REPO, name), encoding="utf-8") as fh:
        return fh.read().lower()


@pytest.mark.parametrize("doc", DOC_TABLES)
def test_docs_name_the_predicate_as_the_source_of_truth(doc):
    assert "render_envelope.py" in _doc_text(doc)


@pytest.mark.parametrize("doc", DOC_TABLES)
def test_docs_state_the_wavefront_only_integrators(doc):
    """Every integrator the predicate refuses under the megakernel must be
    described as wavefront-only in the documented table."""
    text = _doc_text(doc)
    derived = [i for i in re.INTEGRATORS
               if not evaluate(Q(i, "megakernel")).ok]
    assert derived, "no wavefront-only integrator derived — check the predicate"
    for integ in derived:
        assert f"{integ}" in text and "wavefront-only" in text, integ
        # the claim has to sit near the integrator's own name, not anywhere
        assert any("wavefront-only" in line and integ in line
                   for line in text.splitlines()), f"{doc} does not call {integ} wavefront-only"


@pytest.mark.parametrize("doc", DOC_TABLES)
def test_docs_state_the_spectral_exclusions(doc):
    """The axes the predicate refuses under --spectral (neural proposal, ReSTIR
    reuse, non-flat materials) must be stated in the documented spectral scope."""
    text = _doc_text(doc)
    flat = Q("path", "wavefront", spectral=True)
    assert re.SPECTRAL_NO_NEURAL in evaluate(
        Q("path", "wavefront", ("neural",), spectral=True)).codes
    assert re.SPECTRAL_NO_REUSE in evaluate(
        Q("path", "wavefront", (), "restir-di", spectral=True)).codes
    assert re.SPECTRAL_FLAT_ONLY in evaluate(
        Q("path", "wavefront", spectral=True, material_class="skin")).codes
    assert evaluate(flat).ok, "spectral path/wavefront/flat should be in-envelope"
    # Each derived exclusion must be stated on a line that also says "spectral" —
    # several phrasings are accepted, since this guards against forgotten docs,
    # not against rewording.
    facts = {
        "the neural proposal is refused": ("neural proposal",),
        "ReSTIR reuse is refused": ("restir reuse", "restir di reuse"),
        "materials are flat-only": ("flat materials only", "flat-material only",
                                    "flat material only", "flat materials in v1"),
    }
    lines = [ln for ln in text.splitlines() if "spectral" in ln]
    for fact, phrasings in facts.items():
        assert any(p in ln for ln in lines for p in phrasings), \
            f"{doc} does not state, on a spectral line, that {fact}"


if __name__ == "__main__":  # regeneration entry point
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w", encoding="utf-8") as fh:
        json.dump(_intern(build_snapshot()), fh, indent=0, sort_keys=True)
        fh.write("\n")
    print(f"wrote {FIXTURE}")
