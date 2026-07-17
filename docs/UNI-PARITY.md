# UNI-PARITY — uni-api gateway vs heaven `run_langchain` vs heaven `run_on_uni_api`

**The parity ledger.** Three columns: what the uni-api *gateway* supports (yym68686/uni-api +
uni-api-core, surveyed 2026-07-10), what heaven's `run_langchain` does (the maintained reference
implementation), and what heaven's `run_on_uni_api` actually does (the active version at
`baseheavenagent.py` ~L3913 as of 2026-07-10 — the commented-out v1 that sat above it was
deleted in the 2026-07-10 parity commit).

**Rule: any commit that touches `run()`, `run_langchain`, `run_on_uni_api`, `_execute_tool_calls_uni`,
`_prepare_tools_for_uni_api`, or `History.to_uni_messages`/`from_uni_messages` updates this table in
the SAME commit.** Mark in-code divergences with a `# PARITY:` comment pointing here.
Line numbers refer to the 4615-line baseheavenagent.py as of 2026-07-10 and will drift.

Status key: ✅ works · ❌ missing · 🐛 bug · ⚠️ partial/divergent · ➖ not applicable · 🚫 won't-fix (by design)

---

## 1. The core matrix

| # | Capability | uni-api gateway | heaven `run_langchain` | heaven `run_on_uni_api` | Severity / ruling |
|---|---|---|---|---|---|
| 1 | Hook firing: BEFORE_RUN / AFTER_RUN | ➖ client concern | ✅ (L2366, L2748) | ✅ | 🔴 CRIT — fixed 2026-07-10 |
| 2 | Hook firing: BEFORE/AFTER_ITERATION | ➖ | ✅ (L2414, L2739) | ✅ | 🔴 CRIT — fixed 2026-07-10 |
| 3 | Hook firing: BEFORE_TOOL_CALL **(veto-capable — hooks can block tools)** | ➖ | ✅ (L2611, block via `ctx.data["block"]`) | ✅ (in `_execute_tool_calls_uni`, same veto contract) | 🔴 CRIT (safety) — fixed 2026-07-10 |
| 4 | Hook firing: AFTER_TOOL_CALL | ➖ | ✅ (L2633) — fires on errored tools too (execution wrapped in try → `ToolResult(error=...)`, L2649-2656) | ✅ (in `_execute_tool_calls_uni`) — round 3: now ALSO fires on errored tools (execution exceptions become `ToolResult(error=...)` and flow through the same callback+hook+tool-message path; previously the outer except skipped the hook AND tool_output_callback, silently disabling devdir tracking (#6) on errored tools) | 🔴 CRIT (drives #6) — fixed 2026-07-10; errored-tool gap fixed round 3 |
| 5 | Hook firing: ON_ERROR | ➖ | ✅ (L2783) | ✅ | 🔴 CRIT — fixed 2026-07-10 |
| 6 | Devdir/AIOS context: `_track_active_work_dir` → `resolve_devdirs` (rules of dirs the agent reads/bashes into) | ➖ | ✅ (AFTER_TOOL_CALL hook + refresh) | ✅ (AFTER_TOOL_CALL hook + refresh) | 🔴 CRIT — fixed 2026-07-10 |
| 7 | `refresh_system_prompt()` per iteration AND mid-tool-loop **before** showing tool result (twofold devdir mechanism, Isaac 2026-07-09, L2703-2714) | ➖ | ✅ | ✅ (refresh at iteration start + mid-tool-loop, slot-0 sync into both histories) | 🔴 CRIT — fixed 2026-07-10 |
| 8 | Persona directives (`skillmanager_persona=` / `absolute_…`, sticky, via resolve_devdirs) | ➖ | ✅ | ✅ (via #7) | 🔴 CRIT — fixed 2026-07-10 |
| 9 | BEFORE_SYSTEM_PROMPT hook | ➖ | ✅ (inside refresh_system_prompt) | ✅ (rides #7) | 🔴 (rides #7) — fixed 2026-07-10 |
| 10 | Auto-compaction `_maybe_compact` before every model call | ➖ | ✅ (L2471, 2554, 2717) | ✅ (`_maybe_compact_uni`: compacts the langchain mirror, rebuilds uni history in lockstep) | 🔴 CRIT for 1M-ctx M3 — fixed 2026-07-10 |
| 11 | Token-usage → `context_window_config` | ✅ returns `usage` | ➖ (char-count threshold only) | ✅ (`_consume_uni_usage` after every `invoke_uni_api`; no tiktoken fallback) | 🟠 HIGH — fixed 2026-07-10 |
| 12 | Error handling: try/except + history save-on-failure fallback | ➖ | ✅ (save_error path L2766) | ✅ (try/except + ON_ERROR + best-effort save + save_error fallback) | 🟠 HIGH — fixed 2026-07-10 |
| 13 | `_sanitize_history()` (dedupe consecutive HumanMessages) | ➖ | ✅ (L2348) | ✅ (at run start) | 🟡 MED — fixed 2026-07-10 |
| 14 | `_stamp_ts` timestamps (CartON/observatory keys `Iteration_{ts}` off `additional_kwargs.timestamp`) | ➖ | ✅ all messages | ✅ all mirrored langchain messages | 🟠 HIGH — fixed 2026-07-10 |
| 15 | **Images: user-attached** | ✅ full — every provider builder handles `type:"image_url"` (payloads.py L393, 969, 1176, 1351, 1707, 1905, 1990, 2117, 2392; per-provider `get_image_message`; config gate `provider.image` default true) | ✅ `images` param → `_build_human_content` (per-provider shapes, L2315) | ✅ `images` param → `_build_uni_human_content` (ALWAYS OpenAI shape; gateway converts per provider); `run()` forwards on all three uni call sites. Round 3: the LC mirror's HumanMessage gets a `deepcopy` of the block list (was the SAME object as the uni dict's — aliasing: mutation via either side corrupted the other) | 🟠 HIGH — fixed 2026-07-10; aliasing fixed round 3 |
| 16 | **Images: tool-result base64_image** | ✅ (as image_url content on a user turn) | ✅ explicit branches (L2641, 3133, 3172) | ✅ `_execute_tool_calls_uni` emits a follow-up `role:"user"` image_url turn when `tool_result.base64_image`. Round 3: image turns are collected and appended AFTER the batch's LAST `role:"tool"` message (never interleaved between tool messages — invalid OpenAI ordering if a multi-tool batch ever arrives despite the row-20 clamp) | 🟠 HIGH — fixed 2026-07-10; batch ordering hardened round 3 |
| 17 | **Images: history portability** | expects OpenAI shape (`image_url.url`) | produces PROVIDER shapes (Anthropic `type:"image"`, Google data-URI string) | ✅ `to_uni_messages` normalizes HumanMessage list content to OpenAI shape (Anthropic base64 + Google data-URI string → `image_url` dicts; unknown blocks dropped). Round 3 edge cases: Anthropic blocks with missing/empty `source.data` are SKIPPED (were emitting empty data-URIs); if normalization drops every block the message falls back to `content: ""` (empty list 400s at the gateway) | 🟠 HIGH — fixed 2026-07-10; edge cases round 3 |
| 18 | Tool schema conversion | ✅ OpenAI tools format | ✅ per-provider bind_tools (init + re-bind after MCP L2363) | ✅ `_prepare_tools_for_uni_api` (heaven `get_openai_function` + langchain `convert_to_openai_function`) | ✅ parity |
| 19 | MCP tools (`resolve_mcps`) | ➖ | ✅ (L2353) | ✅ (L3946) | ✅ parity |
| 20 | Parallel tool calls | ⚠️ has a multiple-tool-call bug (per heaven comment) → heaven sends `parallel_tool_calls: false` | ✅ executes ALL calls per response + synthetic ToolMessages for orphans/abandoned (MiniMax 400 (2013) fix, L2572, 2667) | ⚠️ single-tool clamp (`_cleanse_dangling_tool_calls` MULTIPLE_TOOL_CALLS) — one call per round-trip, slower. Round 2: the cleanse now mirrors its clamps into the LC history too (was uni-only → lockstep violation, see §5.2) | 🟡 labeled-intentional; revisit if gateway bug fixed |
| 21 | MiniMax XML tool-call fallback nudge | ➖ (envelope normalizes) | ✅ (L2533-2559) | ❌ — presumed N/A via OpenAI envelope; UNVERIFIED against M3-through-uni | 🟡 verify once |
| 22 | DUO sidechain (Challenger injection) | ➖ | ✅ present (L2433) | ❌ (commented v1 had it) | 🚫 **won't-fix — DUO deprecated by design (Isaac 2026-07-10). Label both paths.** |
| 23 | `temperature` param | pass-through | ✅ omitted when None (L1040) | ✅ omitted when None (mirrors run_langchain) | 🔴 BUG — omit when None — fixed 2026-07-10 |
| 24 | `max_tokens` | pass-through | ✅ config value (default 8000) | ✅ config value (getattr fallback 4000 unreachable — attr always exists) | ✅ (misleading fallback; cosmetic) |
| 25 | `thinking_budget` | ➖ | ✅ passed to UnifiedChat.create | ❌ not in payload | 🚫 **not a parity item — providers moved to effort levels (Isaac 2026-07-10). Map a `reasoning_effort` passthrough via #26 instead; label.** |
| 26 | `extra_model_kwargs` | pass-through (arbitrary payload keys) | ✅ merged into model_params (L1042) | ✅ merged into payload. Round 3: the `"model"` key is EXCLUDED from the merge — `invoke_uni_api(model=..., **payload)` raises TypeError on collision; model is fixed by `config.model` (override via config, not extra_model_kwargs) | 🔴 **CRIT (Isaac ruling 2026-07-10)** — fixed 2026-07-10; this is also the vehicle for effort-level params |
| 27 | Agent mode (goal/iterations, `_format_agent_prompt`, TaskSystemTool, `additional_kws` extraction) | ➖ | ✅ | ✅ (shared helpers). Round 2: detect-before-append (raw `agent goal=…` command no longer stored as a user turn), agent-command detection on the trailing history user turn, continuation gating (`goal or continuation_prompt`) + `continuation_iterations` override — see §5.5/§5.6 | ✅ parity — fixed 2026-07-10 round 2 |
| 28 | Block report (WriteBlockReportTool → `create_block_report` → ON_BLOCK_REPORT hook) | ➖ | ✅ (keys off canonical `tool.name`) | ✅ — and uni ADDITIONALLY auto-injects the "I've created a block report…" assistant turn. ON_BLOCK_REPORT fires on both (inside create_block_report). Round 2: a blocked iteration now exits BEFORE AFTER_ITERATION fires and without incrementing current_iteration (mirrors reference; was firing both). Round 3: detection is case-INSENSITIVE (`tool_name.lower()`) — tool lookup matches `.lower()`, so a differently-cased model call executed WITHOUT setting `blocked`; same fix applied to the TaskSystemTool check (identical bug class, same matching contract) | ✅ (uni slightly ahead here) — fixed 2026-07-10 rounds 2-3 |
| 29 | SM cycle enforcement (`min_sm_cycles`) | ➖ | ✅ (lives in `run()` dispatcher) | ✅ (same) | ✅ parity |
| 30 | Streaming | ✅ SSE supported | ❌ non-streaming `ainvoke` + callback simulation | ❌ non-streaming `invoke_uni_api` | 🟡 neither path streams; future item |
| 31 | Return shape | ➖ | `{history, history_id, agent_name, agent_status}` | same + `uni_api_used, context_window_status, raw_response`. Round 2: `raw_response` is now ALWAYS the full gateway response (tool-loop runs previously returned the bare message dict — inconsistent shape). NOTE: no in-repo raw_response consumer found on audit (the AutoSummarizingAgent claim in round 1 was wrong — it uses `look_for_particular_tool_calls`, row 33) | ⚠️ divergent (callers must tolerate extras) — fixed 2026-07-10 round 2 |
| 32 | **Assistant TEXT on tool-call turns persisted to history** | ➖ | ✅ (whole AIMessage appended — never destructured) | ✅ mirrored via `from_uni_messages` (same conversion as the mid-tool-loop append) — was `AIMessage(content="")`: text DROPPED from the LC mirror while the uni side kept it (dual-history divergence + silent history loss) | 🔴 CRIT (reviewer CRITICAL-1) — fixed 2026-07-10 round 2 |
| 33 | `look_for_particular_tool_calls()` post-save subclass hook (AutoSummarizing/PhaseDetector/ConceptResolver/SubphaseDetector) | ➖ | ✅ (after save, L2779) | ✅ (after save, success path) — was never fired on the uni path | 🟠 HIGH — fixed 2026-07-10 round 2 |
| 34 | `_process_agent_response` on every response at arrival (additional_kws/XML extraction in CHAT mode; task bookkeeping on mid-tool-loop text) | ➖ | ✅ arrival-time on every response; PLUS goal-gated re-process of the iteration's FIRST response at end (a double-extraction quirk → duplicate `{kw}_2` entries) | ✅ arrival-time at all three append sites, each response processed exactly ONCE — reference's end-of-iteration re-process intentionally NOT replicated (labeled in-code) | 🟠 HIGH — fixed 2026-07-10 round 2; ⚠️ intentional divergence on the double-process quirk |
| 35 | `heaven_main_callback` coverage (user prompt turn, agent-mode prompt, tool results — always LC message objects) | ➖ | ✅ fires on every appended message | ✅ — was assistant-turns-only, and one mid-loop site passed the raw uni dict instead of the LC message | 🟡 MED — fixed 2026-07-10 round 2 |

## 2. Gateway capabilities heaven doesn't use at all (opportunity list)

| uni-api capability | heaven usage | Note |
|---|---|---|
| Load balancing (weighted / round-robin / key-level) | ❌ | free reliability once uni path is default |
| Channel cooling + auto-retry across channels | ❌ | ditto |
| Per-key rate limits & model permission scoping | ❌ | the SaaS metering seam (avi-jw paid tier) |
| `/v1/images/generations`, `/embeddings`, `/audio/*`, `/moderations`, `/video` | ❌ | gpt-image/video gen behind same envelope |
| **Codex/Claude-Code OAuth channel (`auth/codex_oauth.py`)** | ❌ | relevant to BYOK tier design — investigate ToS posture before use |
| `/v1/models`, health/stats endpoints | ❌ | model discovery + ops |

## 3. Fossils & adjacent breakage (cleanup list)

1. **Commented-out `run_on_uni_api` v1** (L3586–3911, ~325 lines): ✅ **DELETED 2026-07-10.** It *looked*
   more capable than the live one (usage extraction, DUO, error handling) — that camouflage is how
   this parity loss went unnoticed.
2. **`streamlit_run` (L2788): DEPRECATED — DO NOT TOUCH (Isaac ruling 2026-07-10).** Known broken
   (live `f.write` with commented-out file handles → NameError; content destructuring that drops
   tool_use blocks). Ignored entirely; `run_langchain` + callbacks supersedes it. Not part of the
   parity scope.
3. **`run_adk`**: functional but self-flagged ("Idk why this even got in here... it's garbage" L2145;
   history-append bug note L2172; NOTES block L2231). Out of parity scope; label as experimental.
4. `to_uni_messages` also silently drops assistant `thinking` blocks and joins text parts — fine,
   but label it.
5. `from_uni_messages` (round 2): malformed tool-call `arguments` JSON no longer crashes history
   conversion — falls back to `{"__raw_arguments__": <raw string>}`. This converter now runs on
   run_on_uni_api's hot mirror path (every assistant tool-call turn, per matrix row 32), so a
   single bad-args model turn must not kill the run.
6. `to_uni_messages` emits the AIMessage `additional_kwargs` dict INTO the uni message dicts
   (it is the from/to round-trip vehicle) — meaning `invoke_uni_api` sends that non-OpenAI key
   (now carrying row-14 timestamps) to the gateway on bootstrap/compaction-rebuilt histories.
   Pre-existing (pre-round-1) exposure; gateway currently tolerates it. Flagged round 2, NOT
   changed — candidates: strip at the invoke seam. Revisit if the gateway starts validating.
7. **Pre-existing, out of scope, noted (round 3):** `run()`'s streamlit dispatch (~L2306) calls
   `self.streamlit_run(prompt, output_callback, tool_output_callback)` — but `streamlit_run`'s
   signature is `(output_callback, tool_output_callback, heaven_main_callback=None, prompt=None)`,
   so `prompt` lands in the `output_callback` positional slot (and every arg after shifts).
   DO NOT FIX — `streamlit_run` is deprecated + do-not-touch (§3.2); the broken dispatch is
   part of the same fossil.
8. **Pre-existing, out of scope, noted (round 3):** `run_on_uni_api`'s slot-0 system-prompt sync
   (`_sync_system_slot` + bootstrap) assumes `to_uni_messages()` output is index-aligned with
   `history.messages` — but `to_uni_messages` SKIPS ADK events, so a history containing ADK
   events would misalign the two lists from bootstrap on (exotic: uni + ADK histories don't
   mix in practice). Noted, not changed.

## 4. Fix plan (agreed order)

1. Hooks: fire BEFORE/AFTER_RUN, BEFORE/AFTER_ITERATION, BEFORE_TOOL_CALL (with veto → ToolResult
   error, mirroring L2615-17), AFTER_TOOL_CALL, ON_ERROR in the uni loop.
2. `refresh_system_prompt()` at iteration start AND mid-tool-loop before the post-tool model call
   (sync into `uni_conversation_history[0]` + langchain mirror).
3. `_maybe_compact` on the langchain mirror before each `invoke_uni_api`; restore
   `context_window_config.update_from_uni_api(result["usage"])`.
4. try/except around the loop: ON_ERROR hook + save-history fallback (mirror L2766-2779).
5. `_stamp_ts`-equivalent timestamps on every mirrored langchain message (observatory identity).
6. `temperature`: include only when `config.temperature is not None`.
7. **`extra_model_kwargs`: merge into payload** (CRIT).
8. Images (three parts): `images` param plumbed through `run()` → uni; a uni variant of
   `_build_human_content` that ALWAYS emits OpenAI shape (the gateway does per-provider conversion —
   stop pre-shaping); `_execute_tool_calls_uni` emits follow-up user-role image message when
   `tool_result.base64_image`; `to_uni_messages` normalizes Anthropic/Google image blocks →
   OpenAI `image_url` data-URIs.
9. `_sanitize_history()` at run start.
10. Labels: `# PARITY:` comments at every remaining intentional divergence (single-tool clamp,
    DUO-deprecated, thinking_budget→effort-levels note); docstring on each run method pointing here.
11. Delete fossil v1. (`streamlit_run`: deprecated + ignored — out of scope, do not modify.)

---

## 5. Fix round 2 (2026-07-10, post-review — reviewer FAILed round 1 `2ddd59c`)

Independent review of the round-1 commit returned FAIL. The transmitted issue list was lost
except item 1's title; items 2–9 below were reconstructed by a full re-audit of
`run_on_uni_api` against the `run_langchain` reference. All fixed in this commit:

1. **CRITICAL-1 — assistant text dropped from the LC mirror on tool-call turns** (row 32).
   The tool-call-branch mirror was `AIMessage(content="")`; now mirrors via
   `from_uni_messages([assistant_message])` — same conversion as the mid-tool-loop append —
   preserving text + both tool_calls representations (additional_kwargs OpenAI-format AND
   LC-native `tool_calls`).
2. **`_cleanse_dangling_tool_calls` lockstep** (row 20). Accepted the LC history and ignored
   it; both the MULTIPLE_TOOL_CALLS clamp and the MAX_TOOL_CALLS strip now mirror into the LC
   last message (preserving non-tool_calls additional_kwargs, i.e. row-14 timestamps).
   Pre-mirror call sites pass `[]` (uni-only clamp, nothing to mirror yet).
3. **`look_for_particular_tool_calls()`** (row 33) — post-save subclass hook, never fired on uni.
4. **`_process_agent_response` arrival parity** (row 34) — chat-mode additional_kws extraction
   and mid-tool-loop task bookkeeping were lost (only goal-gated, last-message-only before).
5. **Agent-command prompt handling** (row 27) — detect BEFORE append; raw `agent goal=…`
   command no longer stored (was creating consecutive user turns: raw command + formatted
   agent prompt); detection on the trailing history user turn (prompt-less continuations).
6. **Continuations** (row 27) — `continuation_iterations` overrides max_iterations;
   agent-prompt gating is `goal or continuation_prompt`. (Reference's dead
   `self.current_iterations = 1` typo intentionally not replicated — labeled in-code.)
7. **`heaven_main_callback` coverage** (row 35) — user/agent-prompt/tool turns now included;
   raw-dict leak at one mid-loop site fixed (always LC objects).
8. **`raw_response` shape** (row 31) — always the full gateway response.
9. **Blocked-iteration hook semantics** (row 28) — exit before AFTER_ITERATION, no increment.

Plus §3.5 (`from_uni_messages` malformed-args hardening — now load-bearing via item 1) and
§3.6 (flagged-not-fixed: additional_kwargs leak to the wire, pre-existing).

Verified by standalone unit exercise of `from_uni_messages` (text+tool_calls mirror, malformed
args, uni round-trip) and `_cleanse_dangling_tool_calls` (both clamp modes, lockstep, empty-LC
call sites). Full-framework import untestable in this container (langchain_anthropic /
langgraph version drift — pre-existing, unrelated).

---

## 6. Fix round 3 (2026-07-10, final — remaining reviewer items)

Rounds 1-2 fixed the rest of the reviewer's report; these are the remaining items, all fixed
in this commit:

1. **AFTER_TOOL_CALL fires on errored tools** (row 4). `_execute_tool_calls_uni` tool
   execution is now wrapped per-branch in try → `ToolResult(error=str(e))` (mirrors
   `run_langchain` L2649-2656), so execution exceptions flow through the SAME
   tool_output_callback + AFTER_TOOL_CALL + tool-message path as success. Previously the
   outer except appended the error tool message but skipped the hook and callback —
   silently disabling `_track_active_work_dir` devdir tracking on errored tools. The outer
   except remains as a last-resort guard for malformed tool_call envelopes only (no tool
   ran → no hook; the id still gets its role:"tool" message).
2. **Image follow-up batch ordering** (row 16). Tool-result image user turns are collected
   during the loop and appended after ALL role:"tool" messages of the batch — never
   interleaved (invalid OpenAI ordering if a multi-tool batch ever arrives despite the
   row-20 single-tool clamp).
3. **Case-insensitive special-tool detection** (row 28). `WriteBlockReportTool` checks (both
   `_execute_tool_calls_uni` and run_on_uni_api's blocked-response injection, which reads
   the model-cased `tm["name"]`) now compare via `.lower() == "writeblockreporttool"` —
   tool lookup matches `.lower()`, so a differently-cased call executed without setting
   `blocked`. Same fix applied to the adjacent `TaskSystemTool` check (identical bug class).
4. **Multimodal content aliasing** (row 15). The user-turn block list from
   `_build_uni_human_content` was the SAME object in the uni dict and the mirrored
   HumanMessage; the mirror now gets a `deepcopy`.
5. **`to_uni_messages` normalization edge cases** (row 17). Anthropic image blocks with
   missing/empty `source.data` are skipped (were emitting empty data-URIs); all-blocks-
   dropped falls back to `content: ""` (empty list → gateway 400).
6. **`extra_model_kwargs` "model" key collision** (row 26). Excluded from the payload merge —
   `invoke_uni_api(model=..., **payload)` would raise TypeError; model is fixed by
   `config.model`.
7. Ledger truth pass: rows 4/15/16/17/26/28 updated; §3.7 (streamlit dispatch positional-arg
   fossil) and §3.8 (slot-0 sync vs ADK-event index alignment) recorded as pre-existing,
   out of scope.

Verified by `py_compile` on both files and a standalone stub-agent exercise of
`_execute_tool_calls_uni`: raising tool → AFTER_TOOL_CALL fired with `ToolResult.error`,
tool_output_callback fired, valid role:"tool" message emitted; image follow-up ordering
checked with a two-tool batch (user turn after both tool messages); case-insensitive
detection checked with a lowercased WriteBlockReportTool call.

---
*Created 2026-07-10 (Isaac + Claude session in the avi-jw dev container) from a full read of
baseheavenagent.py (4615 lines), history.py to/from_uni_messages, and the uni-api + uni-api-core
sources at HEAD. Fix round 2 applied 2026-07-10 (see §5); fix round 3 applied 2026-07-10 (see §6).*
