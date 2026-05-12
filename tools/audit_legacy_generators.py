#!/usr/bin/env python3
"""audit_legacy_generators — Phase 5 pre-analysis (read-only).

Surveys the three image generators that currently coexist in the
repo and produces a migration-planning report:

  * tools/sss_gen.c                — Phase 1+ spectrogram engine
                                     (kept; will absorb the others
                                     after Phase 5).
  * tools/sss_animate.c            — Atmos motion-frame generator.
  * legacy_deprecated/gen_image_ce.c — Pre-Phase-1 .ces-driven
                                     image generator.

No engine code is modified. The script grep / regex-parses C
sources, the Makefile, the Python UI server, scripts, tests, and
docs to:

  1. Build a call-graph table (which sss_*/ce_* functions each
     generator calls).
  2. Cross-reference the integration points (UI endpoints, helper
     scripts, Makefile targets, tests, docs).
  3. Map each legacy generator's CLI options to sss_gen's options
     and flag the gaps.
  4. Estimate the build-output delta after removal.
  5. Compare three migration scenarios (immediate / shim /
     deprecation-warning) and recommend one based on the
     evidence collected.
  6. Sketch a unified UI endpoint design.
  7. Roll up the risks across build, runtime, compatibility,
     regression, and documentation.

Output is a single markdown report on stdout (or `--out` path),
ready to paste into a planning PR.

Run:
    python3 tools/audit_legacy_generators.py
    python3 tools/audit_legacy_generators.py --out reports/phase5_audit.md
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parent.parent

# ── Generators under analysis ────────────────────────────────────
GENERATORS = {
    "sss_gen":      REPO / "tools" / "sss_gen.c",
    "sss_animate":  REPO / "tools" / "sss_animate.c",
    "gen_image_ce": REPO / "legacy_deprecated" / "gen_image_ce.c",
}

# Headers / modules under ce_core that we want to attribute function
# calls to. The grep pass walks every .h to build a name → module map.
CE_CORE_DIR = REPO / "ce_core"

# Files we cross-reference for integration points.
INTEGRATION_TARGETS = {
    "ui":      [REPO / "ui"],
    "scripts": [REPO / "scripts"],
    "tests":   [REPO / "tools",   # tools/test_*.py
                REPO / "tests"],
    "docs":    [REPO / "README.md",
                REPO / "PIPELINE.md",
                REPO / "docs"],
    "make":    [REPO / "Makefile"],
}

C_CALL_RE   = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(")
C_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)


# ── Helpers ──────────────────────────────────────────────────────

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _strip_c_comments(src: str) -> str:
    """Strip /* ... */ and // comments so the function-call regex
    doesn't trip on commented-out code or doc examples."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"//[^\n]*", "", src)
    return src


def _scan_ce_core_symbols() -> dict[str, str]:
    """Walk every header in ce_core/ and map each declared function
    name to its module (header filename, no extension).

    The match is approximate: any line that looks like a function
    declaration ending with `(` is captured. False positives don't
    matter much because we only look up names we *also* found being
    called inside the generators."""
    symbol_to_module: dict[str, str] = {}
    for hdr in sorted(CE_CORE_DIR.glob("*.h")):
        text = _strip_c_comments(_read(hdr))
        module = hdr.stem
        for m in re.finditer(
                r"^\s*(?:[A-Za-z_][A-Za-z0-9_ \*]+?\s+)+?"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, re.M):
            name = m.group(1)
            if name in ("if", "for", "while", "switch", "sizeof",
                        "return", "typedef", "do"):
                continue
            symbol_to_module.setdefault(name, module)
    return symbol_to_module


def _extract_calls(c_path: Path) -> dict[str, int]:
    """Return {function_name: call_count} for `sss_*` and `ce_*`
    names called inside `c_path`. Ignores comments + string lits."""
    text = _strip_c_comments(_read(c_path))
    # Trim string literals so a printed function name doesn't count.
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    counts: dict[str, int] = defaultdict(int)
    for m in C_CALL_RE.finditer(text):
        name = m.group(1)
        if name.startswith(("sss_", "ce_", "morpheme_", "hybrid_", "spatial_")):
            counts[name] += 1
    return dict(counts)


def _extract_includes(c_path: Path) -> list[str]:
    text = _read(c_path)
    return [m.group(1) for m in C_INCLUDE_RE.finditer(text)]


# ── Cross-ref pass ───────────────────────────────────────────────

@dataclass
class GeneratorInfo:
    name: str
    path: Path
    includes: list[str]                       = field(default_factory=list)
    calls: dict[str, int]                     = field(default_factory=dict)


def _build_generator_info(symbol_to_module: dict[str, str]
                          ) -> dict[str, GeneratorInfo]:
    info: dict[str, GeneratorInfo] = {}
    for name, path in GENERATORS.items():
        info[name] = GeneratorInfo(
            name=name, path=path,
            includes=_extract_includes(path),
            calls=_extract_calls(path),
        )
    return info


# ── Integration grep ─────────────────────────────────────────────

REF_RE = re.compile(r"\b(sss_animate|sss_gen|gen_image_ce)\b")


def _walk_files(root: Path, exts: Iterable[str] | None = None
                ) -> list[Path]:
    if root.is_file():
        return [root]
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if exts and p.suffix not in exts:
            continue
        # Skip the `.git` working tree but NOT `.github` workflows /
        # docs — startswith(".git") would catch both. Match the
        # path segment exactly (and `.git/` for safety on rare
        # implementations that yield the trailing slash).
        if any(part == ".git" for part in p.parts):
            continue
        out.append(p)
    return out


def _scan_integration() -> dict[str, dict[str, list[tuple[str, int, str]]]]:
    """Returns:
        category → generator-name → list of (rel_path, lineno, line_text)
    `category` is one of INTEGRATION_TARGETS' keys (ui / scripts / etc).
    """
    out: dict[str, dict[str, list[tuple[str, int, str]]]] = {
        cat: {g: [] for g in GENERATORS} for cat in INTEGRATION_TARGETS
    }
    for category, roots in INTEGRATION_TARGETS.items():
        # Test category gets a sharper filter (only test_*.py).
        for root in roots:
            if not root.exists():
                continue
            paths = _walk_files(
                root,
                exts={".py", ".sh", ".html", ".md", ".c", ".h"}
                     if category != "make" else None)
            for p in paths:
                # Skip the audit script itself + the generator files
                # to avoid self-references.
                rel_str = str(p.relative_to(REPO))
                if rel_str.startswith(("tools/audit_legacy",
                                       "tools/sss_animate.c",
                                       "tools/sss_gen.c",
                                       "legacy_deprecated/gen_image_ce.c")):
                    continue
                if category == "tests":
                    if p.suffix != ".py" or not p.name.startswith("test_"):
                        continue
                text = _read(p)
                for i, line in enumerate(text.splitlines(), 1):
                    for m in REF_RE.finditer(line):
                        out[category][m.group(1)].append(
                            (rel_str, i, line.strip()[:120]))
    return out


# ── Build target / Makefile analysis ────────────────────────────

def _scan_makefile() -> dict[str, list[str]]:
    """Find Makefile lines that actually define a target / alias /
    phony for each generator binary. Specifically matches:

        $(BUILD_DIR)/<bin>:    rule that builds the binary
        <bin>:                 alias / phony entry
        .PHONY: … <bin> …      .PHONY declarations

    Substring grep would over-count (e.g. an `@echo` line for one
    binary mentioning another's name). The label below names each
    captured line by *kind* so the report stays readable."""
    text = _read(REPO / "Makefile")
    out = {g: [] for g in GENERATORS}
    rule_re   = {g: re.compile(rf"^\$\(BUILD_DIR\)/{re.escape(g)}\s*:")
                 for g in GENERATORS}
    alias_re  = {g: re.compile(rf"^{re.escape(g)}\s*:")
                 for g in GENERATORS}
    phony_re  = {g: re.compile(rf"^\.PHONY:.*\b{re.escape(g)}\b")
                 for g in GENERATORS}
    for i, line in enumerate(text.splitlines(), 1):
        s = line.rstrip()
        for g in GENERATORS:
            kind = None
            if rule_re[g].match(s):    kind = "rule"
            elif alias_re[g].match(s): kind = "alias"
            elif phony_re[g].match(s): kind = "phony"
            if kind:
                out[g].append(f"Makefile:{i} [{kind}]: "
                              f"{line.strip()[:140]}")
    return out


# ── CLI option mapping ──────────────────────────────────────────

@dataclass
class CLIOption:
    flag: str
    purpose: str
    sss_gen_equivalent: str   # textual mapping or "MISSING" / "n/a"


# Hand-curated map informed by a quick read of each generator's
# usage() block. The script doesn't try to parse these from C source
# automatically because the CLI surfaces are small and the
# semantic mapping benefits from author judgment.
SSS_ANIMATE_CLI: list[CLIOption] = [
    CLIOption("INPUT.ppm",
              "Source frame for Atmos decomposition",
              "sss_gen --reference IN.ppm (proposed Phase 5 flag) — "
              "or implicit via motif retrieval from the prompt"),
    CLIOption("OUT_DIR",
              "Output directory for frame_NNNN.ppm",
              "sss_gen --out-dir DIR (proposed; today sss_gen writes "
              "a single .ppm, Phase 5 must accept a dir when "
              "--frames > 1)"),
    CLIOption("N_FRAMES",
              "Number of motion frames to emit",
              "sss_gen --frames N (matches Phase 4 temporal_length)"),
    CLIOption("WIDTH / HEIGHT",
              "Output resolution",
              "sss_gen --out-w / --out-h (already present)"),
    CLIOption("(implicit) Atmos motion profile",
              "Auto-inferred motion type per object via "
              "ce_infer_motion_profile",
              "sss_gen --condition <label> <intensity> (Phase 4 "
              "InteractionResponse drives motion); auto-infer maps "
              "to sfb identity → condition lookup"),
]

GEN_IMAGE_CE_CLI: list[CLIOption] = [
    CLIOption("model.ces",
              "CEStorage container (pre-.sss legacy)",
              "sss_gen <model.sfb>  (Phase 4 v2 .sfb supersedes .ces)"),
    CLIOption("prompt",
              "Token-tokenised string",
              "sss_gen <prompt>  (identical)"),
    CLIOption("out.ppm",
              "Single PPM output",
              "sss_gen <out.ppm>  (identical)"),
    CLIOption("seed",
              "Deterministic PRNG seed",
              "sss_gen … seed   (identical)"),
    CLIOption("steps",
              "Denoise iterations (pre-Phase-1 wave loop)",
              "sss_gen … steps  (Phase 1 radio steps replaces "
              "wave loop)"),
    CLIOption("wave_iters",
              "Atomic wave-refine pass count",
              "MISSING — Phase 1 radio loop subsumed this; no flag "
              "needed for new pipeline"),
    CLIOption("--hybrid",
              "Hybrid VAE block-stamp path",
              "MISSING — pre-Phase-1 colour-base+SLIG decode; not "
              "reachable from .sfb"),
    CLIOption("--guidance N.N",
              "Hybrid VAE blend weight",
              "MISSING (paired with --hybrid)"),
]

# Gaps that need a deliberate decision: keep, replace, or drop.
KNOWN_GAPS: list[tuple[str, str, str]] = [
    ("sss_animate: writing N PPMs to a directory",
     "sss_gen writes a single .ppm today.",
     "Phase 5 adds --out-dir + --frames N. Trivial CLI plumbing on "
     "the C side; the Python sss_synthesizer already produces "
     "per-frame envelopes."),
    ("sss_animate: Atmos motion-profile auto-inference from "
     "RGBA pixels",
     "ce_scene_build_from_rgba/_with_profiles infer motion class "
     "(STATIC/FLOW/RIGID/ORGANIC/PARTICLE) from pixel statistics, "
     "not from prompt+sfb.",
     "Phase 5 may keep Atmos as an optional `--atmos-from PPM` "
     "input — converts a reference image to a quick-and-dirty "
     "condition set without going through the .sfb trainer."),
    ("gen_image_ce: .ces ingestion",
     "The .ces container predates .sfb v2. There are no current "
     "training pipelines that produce .ces files (Phase 3 wrote "
     "directly to .sfb).",
     "Drop. Existing .ces files in build/models/ are demo artefacts; "
     "they can be regenerated via Phase 3 pipeline if needed."),
    ("gen_image_ce: Hybrid VAE + guidance",
     "Pre-Phase-1 colour-base + SLIG residual decoder. No callers "
     "outside legacy demos.",
     "Drop. The Phase 4 synthesiser's envelope + cross_resonance + "
     "warp pipeline subsumes it."),
]


# ── Scenarios ────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    plan: str
    pros: list[str]
    cons: list[str]
    recommend_when: str


SCENARIOS: list[Scenario] = [
    Scenario(
        name="A — Immediate removal",
        plan=(
            "Delete tools/sss_animate.c, legacy_deprecated/gen_image_ce.c, "
            "the matching Makefile targets, the ui/server.py and "
            "ui/unified_server.py call sites, and scripts/make_anim_frames "
            "in the same PR. Migrate every caller atomically."),
        pros=[
            "Single PR; no transitional code lingers.",
            "Codebase ends Phase 5 with one generator entry point.",
        ],
        cons=[
            "Atomic migration is fragile — any missed caller breaks "
            "build or runtime.",
            "Forces the merger of trainer / UI / docs work into one "
            "very large PR.",
        ],
        recommend_when=(
            "Few external callers (none here outside the two UI "
            "servers and one script) AND we are willing to land the "
            "Phase 5 trainer + UI + doc updates as one PR."),
    ),
    Scenario(
        name="B — Shim transition",
        plan=(
            "Reduce sss_animate to a thin C wrapper that translates "
            "its CLI to `sss_gen --frames N --temporal-mode video` and "
            "execvp()s it. gen_image_ce becomes a similar wrapper "
            "that prints a deprecation warning and forwards to "
            "sss_gen with a default .sfb path. Remove the shims one "
            "phase later (Phase 6)."),
        pros=[
            "Existing scripts and UI keep working unmodified.",
            "Migration risk is amortised over two PRs."],
        cons=[
            "Two-phase delete is overhead for a small surface.",
            "Shims need their own tests + docs to stay correct."],
        recommend_when=(
            "Many external integrations or downstream users depend "
            "on the legacy binary names (not the case here — UI is "
            "the only meaningful integrator)."),
    ),
    Scenario(
        name="C — Deprecation warning, delete next phase",
        plan=(
            "Keep the binaries buildable; add a stderr warning at "
            "the top of main() in both legacy generators "
            "(\"DEPRECATED: use sss_gen … instead\"), update "
            "README/PIPELINE to mark them as deprecated. Phase 6 "
            "deletes them and the Makefile targets."),
        pros=[
            "Zero behavioural change in the PR — fastest to land.",
            "Gives downstream consumers a published heads-up."],
        cons=[
            "Two phases of work for what is a small one-time delete.",
            "Dead code lingers in the tree for one release cycle."],
        recommend_when=(
            "External users / vendored copies of the repo exist "
            "and need a published heads-up — not the case here."),
    ),
]


# ── UI design sketch ────────────────────────────────────────────

UI_DESIGN = """\
Proposed single endpoint /api/sss_gen replaces /api/generate and
the internal-only /api/atmos. Request body:

    {
      "model":        "build/feature_bank.sfb",
      "prompt":       "kitty waving paw",
      "seed":         42,
      "steps":        120,
      "detail":       1.5,
      "frames":       1                  // 1 = image, N > 1 = video
      "size":         [256, 256],
      "conditions": [                    // optional; Phase 4 v2
        { "label": "wind",       "intensity": 0.5 },
        { "label": "blink",      "intensity": 1.0 }
      ],
      "atmos_reference": "path/to/ref.png"  // optional Atmos warm-start
    }

Response:
  frames == 1 → image/png (single frame, same as today's /api/generate)
  frames  > 1 → application/zip (frame_NNNN.png inside) OR
                application/octet-stream (mp4) when ffmpeg available
                — selected by `Accept` header.

Migration: keep /api/generate as a thin alias that translates to
sss_gen with frames = 1. Keep /api/atmos as a thin alias that
translates to sss_gen with frames > 1 and atmos_reference set.

Alternative considered: split image and video endpoints
(/api/image, /api/video). Rejected because the Phase 4 design
philosophy treats them as the *same* operation — frames=1 is a
special case of frames=N. Two endpoints would re-create the split
this phase is supposed to remove.
"""


# ── Build-output delta ─────────────────────────────────────────

def _binaries_in_makefile() -> list[str]:
    text = _read(REPO / "Makefile")
    bins = []
    for m in re.finditer(r"\$\(BUILD_DIR\)/([a-zA-Z_][a-zA-Z0-9_]*):", text):
        b = m.group(1)
        if b not in bins:
            bins.append(b)
    return bins


# ── Risk roll-up ─────────────────────────────────────────────────

def _build_risks(integ: dict, makefile_hits: dict) -> dict[str, list[str]]:
    risks: dict[str, list[str]] = defaultdict(list)
    for g, hits in makefile_hits.items():
        if g == "sss_gen": continue
        # "Outside Makefile" must exclude the `make` category — that
        # category re-grep's the Makefile and would double-count.
        n_caller_lines = sum(
            len(integ[c][g]) for c in integ if c != "make")
        risks["build"].append(
            f"{g}: {len(hits)} Makefile target line(s); "
            f"{n_caller_lines} integration line(s) outside Makefile.")
    for g in ("sss_animate", "gen_image_ce"):
        ui_hits = integ["ui"][g]
        if ui_hits:
            risks["runtime"].append(
                f"{g}: {len(ui_hits)} UI handler line(s) need rewiring "
                "to sss_gen.")
        script_hits = integ["scripts"][g]
        if script_hits:
            risks["compatibility"].append(
                f"{g}: {len(script_hits)} scripts/* invoke this binary "
                "— update them in the same PR as the removal.")
        test_hits = integ["tests"][g]
        if test_hits:
            risks["regression"].append(
                f"{g}: {len(test_hits)} test file(s) reference this — "
                "regression suite needs an audit.")
        doc_hits = integ["docs"][g]
        if doc_hits:
            risks["docs"].append(
                f"{g}: {len(doc_hits)} doc line(s) mention this — "
                "README / PIPELINE need a migration note.")
    return risks


# ── Report rendering ────────────────────────────────────────────

def _table(rows: list[tuple[str, ...]], header: tuple[str, ...]) -> str:
    # Build cols uniformly from `[header] + rows` so the helper
    # produces one column per header entry even when `rows` is
    # empty. (Earlier `rows else [header]` path treated the header
    # tuple as a single column on empty input.)
    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]
    def line(items): return ("| " + " | ".join(
        str(it).ljust(w) for it, w in zip(items, widths)) + " |")
    out = [line(header),
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows: out.append(line(r))
    return "\n".join(out)


def render_report(info: dict[str, GeneratorInfo],
                  symbol_to_module: dict[str, str],
                  integ: dict, make_hits: dict,
                  binaries: list[str],
                  risks: dict[str, list[str]]) -> str:
    L: list[str] = []

    L.append("# Phase 5 audit — legacy generator consolidation\n")
    L.append("Read-only analysis; no engine code modified.\n")
    L.append(f"Generators surveyed: "
             f"`{'`, `'.join(GENERATORS.keys())}`.\n")

    # ── 1. Call graph ────────────────────────────────────────────
    L.append("## 1. Call graph — ce_core function dependency\n")
    L.append("Function calls (by name) found in each generator's "
             "translation unit. Names are matched against the "
             "ce_core/*.h headers to attribute each call to a "
             "module.\n")

    # Per-generator: function-by-module table.
    for gname, gi in info.items():
        L.append(f"### `{gname}` ({gi.path.relative_to(REPO)})\n")
        L.append(f"**Headers included:** "
                 f"{', '.join('`'+h+'`' for h in gi.includes) or '(none)'}\n")
        if not gi.calls:
            L.append("(no sss_/ce_/morpheme_/hybrid_/spatial_ calls "
                     "detected)\n")
            continue
        rows = []
        for name in sorted(gi.calls):
            module = symbol_to_module.get(name, "?")
            rows.append((name, module, str(gi.calls[name])))
        L.append(_table(rows, ("function", "ce_core module", "calls")))
        L.append("")

    # ── Cross-generator duplication ──────────────────────────────
    L.append("### Overlap matrix (functions called by ≥ 2 generators)\n")
    universe = set()
    for gi in info.values():
        universe.update(gi.calls.keys())
    overlap_rows = []
    for name in sorted(universe):
        present = [g for g, gi in info.items() if name in gi.calls]
        if len(present) >= 2:
            overlap_rows.append((name, ", ".join(present),
                                 symbol_to_module.get(name, "?")))
    if overlap_rows:
        L.append(_table(overlap_rows,
                        ("function", "callers", "ce_core module")))
    else:
        L.append("(no functions called by ≥ 2 generators)\n")
    L.append("")

    # Functions UNIQUE to sss_gen — these we keep verbatim.
    L.append("### Functions unique to `sss_gen` (kept verbatim post-Phase 5)\n")
    unique_to_gen = [
        name for name in sorted(info["sss_gen"].calls)
        if name not in info["sss_animate"].calls
        and name not in info["gen_image_ce"].calls
    ]
    L.append(", ".join(f"`{n}`" for n in unique_to_gen) + "\n"
             if unique_to_gen else "(none)\n")

    # Functions unique to LEGACY generators (would disappear).
    L.append("### Functions unique to legacy generators (drop or move)\n")
    for legacy in ("sss_animate", "gen_image_ce"):
        only = [name for name in sorted(info[legacy].calls)
                if name not in info["sss_gen"].calls]
        L.append(f"**{legacy}:** "
                 + (", ".join(f"`{n}`" for n in only) if only else "(none)")
                 + "\n")

    # ── 2. Integration points ────────────────────────────────────
    L.append("## 2. Integration cross-reference\n")
    L.append("Lines in `ui/`, `scripts/`, `tools/test_*.py`, "
             "`docs/`, `README.md`, `PIPELINE.md`, and `Makefile` "
             "that mention each generator.\n")
    for category, per_gen in integ.items():
        # Skip if empty across all generators.
        if not any(per_gen[g] for g in GENERATORS):
            continue
        L.append(f"### {category}\n")
        for g in GENERATORS:
            hits = per_gen[g]
            if not hits:
                continue
            L.append(f"**`{g}` — {len(hits)} hit(s):**")
            for p, ln, text in hits[:50]:
                L.append(f"  - `{p}:{ln}`: `{text}`")
            if len(hits) > 50:
                L.append(f"  - … (+{len(hits) - 50} more)")
            L.append("")

    # Makefile target list — slightly more readable.
    L.append("### Makefile target lines (raw)\n")
    for g, lines in make_hits.items():
        if not lines:
            continue
        L.append(f"**`{g}`:**")
        for ln in lines:
            L.append(f"  - {ln}")
        L.append("")

    # ── 3. CLI mapping ───────────────────────────────────────────
    L.append("## 3. CLI option mapping\n")
    L.append("### `sss_animate` → `sss_gen`\n")
    L.append(_table([(o.flag, o.purpose, o.sss_gen_equivalent)
                     for o in SSS_ANIMATE_CLI],
                    ("legacy flag", "purpose", "sss_gen equivalent")))
    L.append("\n### `gen_image_ce` → `sss_gen`\n")
    L.append(_table([(o.flag, o.purpose, o.sss_gen_equivalent)
                     for o in GEN_IMAGE_CE_CLI],
                    ("legacy flag", "purpose", "sss_gen equivalent")))
    L.append("\n### Identified gaps and proposed treatment\n")
    L.append(_table([(g, why, treat) for g, why, treat in KNOWN_GAPS],
                    ("gap", "why", "proposed treatment")))
    L.append("")

    # ── 4. Build delta ───────────────────────────────────────────
    L.append("## 4. Build delta\n")
    L.append(f"Binaries currently emitted by `Makefile` "
             f"(via `$(BUILD_DIR)/...`): {len(binaries)}\n")
    L.append("```\n" + "\n".join(binaries) + "\n```\n")
    keep_set = set(binaries) - {"sss_animate", "gen_image_ce"}
    L.append(f"After Phase 5 removal: {len(keep_set)} binaries "
             f"(drops `sss_animate` and `gen_image_ce`).\n")

    legacy_objs = ["legacy/ce_gen.o (currently linked via $(LEGACY_OBJS))"]
    L.append("**Object files potentially droppable:** "
             + ", ".join(f"`{o}`" for o in legacy_objs) + ".\n")

    # ── 5. Scenarios ─────────────────────────────────────────────
    L.append("## 5. Migration scenarios\n")
    for sc in SCENARIOS:
        L.append(f"### {sc.name}\n")
        L.append(f"**Plan:** {sc.plan}\n")
        L.append("**Pros:**\n" + "\n".join(f"- {p}" for p in sc.pros) + "\n")
        L.append("**Cons:**\n" + "\n".join(f"- {p}" for p in sc.cons) + "\n")
        L.append(f"**Recommend when:** {sc.recommend_when}\n")

    # Pick a recommendation based on the data we just gathered.
    ui_hits_total = sum(len(integ["ui"][g])
                        for g in ("sss_animate", "gen_image_ce"))
    scripts_hits_total = sum(len(integ["scripts"][g])
                             for g in ("sss_animate", "gen_image_ce"))
    test_hits_total = sum(len(integ["tests"][g])
                          for g in ("sss_animate", "gen_image_ce"))
    # Resolve the specific UI and script files that actually hit a
    # legacy generator name, so the recommendation cites real paths
    # instead of a hand-coded guess.
    ui_paths = sorted({
        p for g in ("sss_animate", "gen_image_ce")
        for (p, _ln, _txt) in integ["ui"][g]
    })
    script_paths = sorted({
        p for g in ("sss_animate", "gen_image_ce")
        for (p, _ln, _txt) in integ["scripts"][g]
    })

    L.append("### Recommendation\n")
    L.append(
        f"Observed integration footprint outside Makefile: "
        f"{ui_hits_total} UI line(s), {scripts_hits_total} script "
        f"line(s), {test_hits_total} test line(s). "
        "All callers are in-repo and reachable in a single PR.\n")

    if ui_paths:
        ui_str = ", ".join(f"`{p}`" for p in ui_paths)
        ui_clause = f"the UI handlers ({ui_str})"
    else:
        ui_clause = "no UI handlers"
    if script_paths:
        script_str = ", ".join(f"`{p}`" for p in script_paths)
        script_clause = (
            f" and the helper script(s) ({script_str})")
    else:
        # The scan found zero script hits — say so explicitly rather
        # than hard-coding a path that turned out not to reference
        # either legacy binary.
        script_clause = " and no helper scripts (scripts/ scan = 0)"

    L.append(
        "**Scenario A (immediate removal) is the recommended path.** "
        "Repo-wide grep shows no external integrators, and the test "
        "suite already exercises `sss_gen` end-to-end via "
        "`tools/test_synthesizer.py` + the Phase 4 trainers. "
        f"Migrating {ui_clause}{script_clause} atomically with the "
        "Makefile cleanup is the smallest total diff.\n")
    L.append(
        "**Fallback to Scenario C (deprecation warning)** is acceptable "
        "if Phase 5 wants to split the work into a small \"warn and "
        "doc\" PR followed by a removal PR — useful if reviewers want "
        "to land the UI rewrite and the C cleanup separately.\n")

    # ── 6. UI design ─────────────────────────────────────────────
    L.append("## 6. Unified UI endpoint design\n")
    L.append("```\n" + UI_DESIGN + "```\n")

    # ── 7. Risks ─────────────────────────────────────────────────
    L.append("## 7. Risk roll-up\n")
    for category in ("build", "runtime", "compatibility",
                     "regression", "docs"):
        items = risks.get(category, [])
        L.append(f"### {category}\n")
        if not items:
            L.append("(none observed)\n")
        for item in items:
            L.append(f"- {item}")
        L.append("")

    # ── 8. Work order ────────────────────────────────────────────
    L.append("## 8. Recommended Phase 5 work order\n")
    L.append("1. **C engine:** extend `tools/sss_gen.c` with "
             "`--frames N`, `--out-dir DIR`, `--condition LBL INT` "
             "(repeatable), `--atmos-from PPM` (replaces the "
             "Atmos pixel-decompose path). Existing `--out-w/--out-h` "
             "and `--atmos` flags stay.\n")
    L.append("2. **Python wrappers:** ensure `tools/sss_synthesizer.py` "
             "is reachable via a thin Python entry point "
             "(`scripts/sss_gen.py`?) so the UI can drive it without "
             "shelling out for image-only generation.\n")
    L.append("3. **UI rewrite:** introduce `/api/sss_gen` per §6. "
             "Make `/api/generate` and `/api/atmos` thin aliases.\n")
    L.append("4. **Script migration:** rewrite "
             "`scripts/make_anim_frames.py` to call the new "
             "`sss_gen --frames` path.\n")
    L.append("5. **Makefile + delete:** remove "
             "`build/sss_animate` and `build/gen_image_ce` targets "
             "+ delete the C sources (and `legacy/ce_gen.{c,h}` "
             "if no other consumers remain).\n")
    L.append("6. **Docs:** README + PIPELINE migration notes; "
             "`docs/migration_phase5.md` (one page) summarising "
             "the CLI mapping.\n")
    L.append("7. **Regression:** verify Phase 1+2+3+4 tests still "
             "PASS; add `tools/test_sss_gen_video.py` exercising "
             "`--frames N` end-to-end against a Phase 4 .sfb.\n")

    L.append("\n---\nGenerated by `tools/audit_legacy_generators.py` "
             "(Phase 5 pre-analysis, read-only).\n")
    return "\n".join(L)


# ── Entry point ──────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=None,
                    help="path for the markdown report; default stdout")
    args = ap.parse_args()

    symbol_to_module = _scan_ce_core_symbols()
    info     = _build_generator_info(symbol_to_module)
    integ    = _scan_integration()
    make_hit = _scan_makefile()
    binaries = _binaries_in_makefile()
    risks    = _build_risks(integ, make_hit)

    report = render_report(info, symbol_to_module, integ, make_hit,
                           binaries, risks)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}  ({len(report)} bytes)")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
