# T13 — heaven-framework productionization plan (Fable, 2026-07-05)

> Baseline (recon-verified): canonical = THIS repo @ **0.1.30** = origin = PyPI = the onionmorph
> venv; the monorepo vendored copy was synced UP to 0.1.30 today (sancrev-alpha `5ae5ae6e`) — no
> divergence anywhere. Ledger: sanctuary-revolution-alpha `fable_state_tables.md` T13.

## The headline finding
**`tests/` is EMPTY and no CI workflow runs pytest** (9 workflows = release/badges/issues only).
Every fix below lands WITH tests; item 1 builds the floor they land on.

## Build order

1. **The test floor + CI step.** A real `tests/` suite over the load-bearing surfaces:
   `BEFORE_TOOL_CALL` veto read-back (H1, shipped 0.1.26 — untested), the Contacts switchboard
   (H2, shipped 0.1.26 — untested), skill hooks (`register_skill_hooks` / the
   `BEFORE_SYSTEM_PROMPT` equip-once path), `HeavenAgentConfig` construction, registry ops,
   tool schema generation, `unified_chat` model construction (no network — construction only).
   Plus ONE CI workflow step running pytest on push. No fix merges without the floor.

2. **Skill-isolation (the WGEL §7 live bug — DESIGN DECISION MINE, pending a full read of
   HEAVEN-DOTDIR-SPEC's default-persona intent).** Mechanism (recon, exact):
   `skill_manager/core.py:1309-1327 _activate_default_persona_if_empty` equips the aggregated
   `default` skillset (= ALL loaded skills, 147 live) for ANY new agent_id with empty state →
   a brand-new agent inherits the host's entire skill surface (observed live on `ranger`,
   2026-07-05). Candidate designs to decide between (after the spec read):
   (a) empty-state default = EMPTY + explicit opt-in to the aggregate persona;
   (b) `HeavenAgentConfig`-level isolation knob, default preserving today's behavior;
   (c) the DOTDIR-spec'd per-agent artifact-skillset path (Isaac's WGEL §7 note: "the fix is
   heaven-persona work") made the constructor default.
   The decision must honor the DOTDIR spec's intent, not just kill the symptom.

3. **Noise hygiene.** (a) `tools/redaction_tool.py:15-20` — optional-dep import failure
   currently ERROR-logs with `exc_info` every import; downgrade to a one-time DEBUG/WARNING
   without traceback. (b) the `thinking_budget` UserWarning seen on every OM agent
   construction — NOT emitted by heaven 0.1.30 itself (recon: phrase absent); trace the
   constructor kwarg path in `unified_chat.py:179-245` vs the langchain model constructor and
   pass it provider-correctly so the downstream warning stops.

4. **H3 — HermesTool split** (`HermesDelegateTool` [agent+goal required] +
   `DelegateWithConfigTool` [hermes_config required] over the same `use_hermes`): the OM-level
   fix proven live 2026-06-30 (model-independent optional-arg skipping), upstreamed verbatim.

5. **D4 — `use_hermes` nested-config loader fix** (HEAVEN-DOTDIR-SPEC §14, designed).

6. **D10 — history read-compaction** (DOTDIR §15) — LAST; largest blast radius; may defer to
   its own pass if 1-5 fill the window.

7. **Publish** per the pypi-archaeology discipline FROM THIS REPO (version bump after 1-6's
   final shape; every shipped fix listed in the release notes).

## Explicitly OUT of this pass
- **cave-harness P1/P2/P3/P4** (`_emit_event` MRO · SSE json crash · poll-loop autostart ·
  /events keepalive) — REAL, recorded in onionmorph UPSTREAM-PROPOSALS.md, but they live in
  `~/claude_code/cave-harness`, a different package: the sibling pass after this one.
- Any behavior change to the DOTDIR loading order (autoloader semantics stay).

## Method
Amendment 3b throughout: my file-grain pseudocode per item → pens type it → I read every line,
stamp/rewrite → the item-1 floor must be green before + after each fix → merge → push (T8.2 law).
