# HEAVEN PROMPT CACHING — finding + patch strategy (rule-28 dir-state)

> **Status: DESIGNED, NOT BUILT. Zero lines of the patch below exist in the tree.**
> Owner ruling: `designs/WOOM-MVP-CLOSED-LOOP.md:663` — *"Heaven's anthropic-via-langchain
> prompt-caching unknown = IGNORE UNTIL LAUNCH, then CRITICAL (pre-launch checklist row)."*
> Prior ruling honored here: `.claude/rules/train_doc_closeout_states.md` act 24–25 —
> **"the caching correction (opt-in, not automatic)"** (owner §32 heaven).
> Investigated 2026-07-29. Every claim below carries its `file:line`; nothing inferred.

## 1. GROUND TRUTH (read firsthand, not grepped)

| fact | evidence |
|---|---|
| heaven has **zero** caching of any kind | `grep -rn "cache_control\|cache_creation\|cache_read\|ephemeral" heaven_base/` → **0 hits** |
| the ANTHROPIC provider IS `ChatAnthropic` | `unified_chat.py:103`, returned at `:253` |
| MiniMax rides the SAME class via a base-URL swap | `unified_chat.py:241-243` (`anthropic_api_url = https://api.minimax.io/anthropic`) |
| installed `langchain-anthropic` | **0.3.13** (OM venv) · `langchain-core` 0.3.84 · `anthropic` 0.97.0 |
| 0.3.13 **does** pass `cache_control` through | verified live: a system block with `cache_control` survives `_format_messages` to the wire; hits read back on `usage_metadata["input_token_details"]` → `{'cache_read','cache_creation'}` |
| 0.3.13 has **no** auto-cache field | `'cache_control' in ChatAnthropic.model_fields` → **False** (the `cache` field is LangChain response memoization, unrelated) |
| the system prompt is **re-derived from disk every iteration** | `refresh_system_prompt` `:1961-1986` → `resolve_devdirs` `:1965` → string-compare `:1973` → `history.messages[0]` replaced `:1983` |
| devdir injection is **huge** | `resolve_devdirs :1593-1706` inlines whole `.claude`/`.heaven` file bodies; caps `PER_DEVDIR_FILE_CHARS = 40_000`, `TOTAL_DEVDIR_CHARS = 400_000` (`:179-180`) |
| a devdir file **mutates on every game action** | `cave_teams/rpg/woom_state.py:47` writes `<onion>/.claude/rules/woom_state.md`; `record()` `:73` fires on boot, every `/woom/action`, place saves, automation saves |

## 2. THE FINDING, IN ONE SENTENCE

Heaven's cacheable prefix is enormous (up to 400 KB of devdir text, rebuilt every iteration) —
**and it is invalidated on nearly every turn by exactly one auto-managed file**, `woom_state.md`,
which lives inside the injected set. Caching the system prompt without splitting that file out
would pay the 1.25× write premium repeatedly and read almost nothing.

**Both halves must land together, or the patch is worse than no patch.**

## 3. THE UPGRADE QUESTION — ANSWERED, AND THE ANSWER IS "DON'T"

`langchain-anthropic` 1.x adds automatic caching (top-level `cache_control` at invoke +
`AnthropicPromptCachingMiddleware` with `ttl`/`min_messages_to_cache`). **We are not taking it:**

```
langchain-anthropic 1.5.3 requires langchain-core >=1.5.2,<2.0.0
heaven has langchain-core 0.3.84 · langchain 0.3.28 · langgraph 0.2.60
sanctuary-dna PINS langgraph==0.2.60 + langchain-core==0.3.84 (deliberately — sdna 0.3.6 exists
because a stale build broke on langgraph 1.x)
```

That is an **ecosystem migration** (heaven → sdna → cave → every consumer) with a hostile
history, to obtain a convenience wrapper over the same `cache_control` dict 0.3.13 already
forwards. The manual patch below is ~40 lines and moves no dependency.

## 4. THE PATCH — exact code

### 4a. Config flags (`baseheavenagent.py`, `HeavenAgentConfig`, near `:488`)

```python
    # --- Anthropic prompt caching (OPT-IN, per the 24–25 ruling: never automatic) ---
    prompt_cache: bool = False              # master switch; False = byte-identical to today
    prompt_cache_ttl: str = "5m"            # "5m" (1.25x write) | "1h" (2x write)
    prompt_cache_min_chars: int = 6_000     # skip the breakpoint below ~1-2k tokens (it would
                                            # silently not cache and still cost a write attempt)
```

### 4b. The ONE builder every site routes through (new method on `BaseHeavenAgent`)

```python
    # Devdir files that are REWRITTEN BY THE SYSTEM ITSELF and therefore must never sit inside
    # the cached prefix. Auto-managed rule files only — see cave_teams/rpg/woom_state.py.
    VOLATILE_DEVDIR_FILES = {"woom_state.md"}

    def _prompt_cache_active(self, text: str) -> bool:
        """Cache only when opted in, on Anthropic, and above the minimum cacheable prefix."""
        if not getattr(self.config, "prompt_cache", False):
            return False
        if self.config.provider != ProviderEnum.ANTHROPIC:
            return False
        return len(text) >= getattr(self.config, "prompt_cache_min_chars", 6_000)

    def _system_message(self, text: str, volatile: str = "") -> SystemMessage:
        """Build the system message.

        Uncached (default): one plain string — byte-identical to the pre-patch behaviour.
        Cached: TWO blocks — [0] the stable prefix carrying the cache breakpoint, [1] the
        volatile tail AFTER it. Anthropic caching is a prefix match, so block [1] may change
        every turn without invalidating block [0].
        """
        if not self._prompt_cache_active(text):
            return SystemMessage(content=text + volatile)
        blocks = [{
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral", "ttl": self.config.prompt_cache_ttl},
        }]
        if volatile:
            blocks.append({"type": "text", "text": volatile})   # no cache_control — deliberate
        return SystemMessage(content=blocks)
```

### 4c. `resolve_devdirs` — split volatile out (`:1621-1641`, inside the instruction loop)

Add one list beside `instruction_parts`, and route the auto-managed files into it:

```python
        volatile_parts: list[str] = []          # NEW — beside instruction_parts at :1616
```

then, in the instruction loop right before `instruction_parts.append(...)` (`:1641`):

```python
                    block = f"### [from {path}]\n{content.strip()}"
                    if os.path.basename(path) in self.VOLATILE_DEVDIR_FILES:
                        volatile_parts.append(block)      # after the breakpoint
                    else:
                        total_chars += len(content)
                        instruction_parts.append(block)   # inside the cached prefix
```

and return the volatile text separately rather than concatenating it into `result` (`:1706`).
Minimal-blast-radius option: keep `resolve_devdirs` returning one string and stash the tail on
`self._volatile_devdir_text`, so no caller signature changes.

### 4d. Route the 5 construction sites through the builder

| line | current | becomes |
|---|---|---|
| `:1177` | `self.history.messages[0] = SystemMessage(content=self.config.system_prompt)` | `= self._system_message(self.config.system_prompt, self._volatile_devdir_text)` |
| `:1183` | `insert(0, SystemMessage(content=self.config.system_prompt))` | `insert(0, self._system_message(...))` |
| `:1945` | `else SystemMessage(content=self.config.system_prompt)` | `else self._system_message(...)` |
| `:1983` | `self.history.messages[0] = SystemMessage(content=updated_prompt)` | `= self._system_message(updated_prompt, self._volatile_devdir_text)` |
| `:2394` | `insert(0, SystemMessage(content=self.config.system_prompt))` | `insert(0, self._system_message(...))` |

`:1981` (the orchestrator branch) takes the same treatment with its concatenated string.

### 4e. Observability — without it the patch is unfalsifiable

At the invoke site (`:2494`, `response = _stamp_ts(await self.chat_model.ainvoke(...))`):

```python
            if getattr(self.config, "prompt_cache", False):
                _det = (getattr(response, "usage_metadata", None) or {}).get("input_token_details", {})
                _log.info("prompt_cache: read=%s created=%s uncached=%s",
                          _det.get("cache_read", 0), _det.get("cache_creation", 0),
                          (getattr(response, "usage_metadata", None) or {}).get("input_tokens", 0))
```

## 5. WHY NOT `refresh_system_prompt`'s change-guard ALONE

`:1973` already skips the rewrite when the string is unchanged — but that guard operates on the
**whole** prompt. One byte of `woom_state.md` flips it, replacing `messages[0]` wholesale. The
guard prevents needless object churn; it does **not** protect a cache prefix. The 4c split is
what makes the guard's "unchanged" case the common case.

## 6. RISKS — named before building, each with its check

| risk | why | check |
|---|---|---|
| **tool-set nondeterminism kills everything** | `tools` render BEFORE `system`; `bind_tools` runs every call (`:2383`) after async `resolve_mcps` (`:2373`). If MCP tool ORDER varies run-to-run, the prefix never matches and no system breakpoint can help. | ✅ **PARTIALLY CLEARED 2026-07-29 (no credential needed).** Two `BaseHeavenAgent` constructions with the DEFAULT tool set produced identical order both times: `['WriteBlockReportTool','TaskSystemTool','SkillTool']` (ORDER and SET both identical). **⚠ HONEST LIMIT: tested with `tools=[]` only — the MCP-loaded case is UNTESTED and is where the risk actually lives** (`resolve_mcps` → `MultiServerMCPClient.get_tools()`); that check needs live MCP servers. Baseline viable; re-run before enabling caching on any MCP-bearing agent |
| persona/skill injection is also volatile | `resolve_devdirs` appends `<AVAILABLE_SKILLS>` (`:1700`) and a persona frame; if equipped skills shift per turn these belong in the volatile block too | inspect two consecutive rendered prompts |
| a 400 on the block-list content shape | some providers reject list-form system content | negative-test MiniMax explicitly (it is v1's runtime) |
| 1-hour TTL costs 2× to write | only pays off at ≥3 reads | leave the default at `5m` |

## 7. TEST PLAN (live, in this order)

1. **Mechanism, outside heaven** — two `ChatAnthropic` calls, same >1024-token cached system
   block. Assert call 1 has `cache_creation > 0`, call 2 has `cache_read > 0`. *(Proves the
   0.3.13 path end-to-end. Needs `ANTHROPIC_API_KEY` in `~/system_config.sh`.)*
2. **Tool determinism** — the §6 row-1 check. If it fails, stop and fix that first.
3. **Heaven, cache OFF** — run an agent; assert the request is byte-identical to pre-patch
   (regression floor: `prompt_cache=False` must change nothing).
4. **Heaven, cache ON** — two turns; assert turn 2 reports `cache_read > 0`.
5. **The invalidation test (the one that matters)** — turn 1, then *mutate `woom_state.md`*,
   then turn 2. Assert `cache_read > 0` **still**. Without 4c this is the test that fails.
6. **MiniMax negative** — same agent on a MiniMax model; assert no 400 and honest degradation.

## 7b. ⭐ THE AUTH FINDING (2026-07-29) — AND WHY IT DOES **NOT** MOOT THIS PATCH

**Isaac's `claude setup-token` credential 401s on the raw Messages API** — both as `x-api-key`
(*"invalid x-api-key"*) and as `Authorization: Bearer` + `oauth-2025-04-20` (*"OAuth access token
is invalid"*). Tested live. **This is correct current Anthropic behaviour, not a broken token.**

**Two DIFFERENT Anthropic changes, routinely conflated — keep them apart:**

| mechanism | what | status (2026-07-29) |
|---|---|---|
| **A — AUTH** | subscription OAuth token on the *raw* Messages API | blocked Jan/Feb 2026, hardened ~Apr (a 400 system-prompt gate → flat 401). **Never paused. Still blocked.** |
| **B — BILLING** | Agent SDK / `claude -p` / third-party-apps-on-Agent-SDK split off subscription quota into metered credit | announced 2026-05-14 for 06-15, **PAUSED 2026-06-15/16, still paused** — these still draw the subscription as before |

So a subscription can still be spent programmatically — but **only THROUGH the harness**
(`claude_agent_sdk` / the `claude` CLI), never by hand-rolling the token into your own HTTP client.
Heaven's `ChatAnthropic` is a hand-rolled client ⇒ heaven can never spend the subscription directly.

**THE LANE THAT ALREADY EXISTS:** `application/gnosys-claude-p/` (BUILT+VERIFIED GREEN, `c8dc0af`;
train §5 names it the intended L2 GNOSYS runtime). `server/p_main_agent.py` is
**`claude_agent_sdk`-backed** (`query` + `ClaudeAgentOptions`, `setting_sources=["project"]`,
`system_prompt={"type":"preset","preset":"claude_code","append":…}`), and its `_provider_env()`
(`:114-124`) **already documents the subscription fallback**: *"Returns {} if NO token is
configured — then the turn uses whatever os.environ already provides (e.g. an OAuth
subscription)."* Pointing it at the subscription = unset `MINIMAX_API_KEY`/`ANTHROPIC_AUTH_TOKEN`,
set `CLAUDE_CODE_OAUTH_TOKEN`, set `DEFAULT_CLAUDE_CODE_MODEL`. **UNVERIFIED — not yet run.**

**⇒ TWO RUNTIME LANES, AND THIS PATCH SERVES THE SECOND:**
1. **harness lane** (gnosys-claude-p → Agent SDK → subscription): Claude Code does its OWN prompt
   caching internally. **This patch is irrelevant there — do not apply it.**
2. **direct-API lane** (heaven `ChatAnthropic` → MiniMax today, Console-key Anthropic later):
   no caching exists, and **this is v1's actual runtime** (MiniMax-only ruling). MiniMax's
   Anthropic-compatible endpoint documents explicit `cache_control` support, so the patch pays off
   on the v1 lane **without any Anthropic credential at all**.

**The patch is therefore NOT blocked on the credential question.** Only step 1 of §7 (the
mechanism proof against Anthropic proper) needs a Console API key; steps 3–6 can run on MiniMax.

## 8. RESIDUE / OPEN

- **Not built.** No code written. This file is the design only.
- Second breakpoint on the growing message tail (to cache conversation history, not just the
  system prefix) — deliberately out of scope for v1; max 4 breakpoints total.
- The 20-block lookback window matters for long tool-call turns; unexamined here.
- Train §32 (heaven) gains a caching clause **only when the patch lands**, carrying its
  consumption status per the §90′ knob-law postmortem — never in the present tense while unbuilt.
