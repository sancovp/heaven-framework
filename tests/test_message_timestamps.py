"""Per-message happen-time stamping (_stamp_ts) survives a real History save -> load round-trip.

LangChain messages carry no native timestamp; heaven stamps it into additional_kwargs at
receipt/creation. History must round-trip additional_kwargs for BOTH Human and AI messages (AI already
did; Human's dict-load paths were dropping it). This is the datetime identity carton/HS keys off.
"""
import os

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from heaven_base.baseheavenagent import _stamp_ts
from heaven_base.memory.history import History
from heaven_base.memory.chat_branching import fork_history


def test_stamp_ts_sets_and_never_overwrites():
    h = _stamp_ts(HumanMessage(content="hi"))
    assert h.additional_kwargs.get("timestamp"), "human message not stamped"
    first = h.additional_kwargs["timestamp"]
    _stamp_ts(h)  # setdefault — must NOT overwrite
    assert h.additional_kwargs["timestamp"] == first


def test_timestamp_survives_history_roundtrip(tmp_path, monkeypatch):
    """Canonical path: History.save -> load_from_id -> from_json, for ALL four message types."""
    monkeypatch.setenv("HEAVEN_DATA_DIR", str(tmp_path))

    s = _stamp_ts(SystemMessage(content="you are terse"))
    h = _stamp_ts(HumanMessage(content="what time is it"))
    a = _stamp_ts(AIMessage(content="now"))
    t = _stamp_ts(ToolMessage(content="result", tool_call_id="tc1"))
    stamped = {s.additional_kwargs["timestamp"], h.additional_kwargs["timestamp"],
               a.additional_kwargs["timestamp"], t.additional_kwargs["timestamp"]}
    assert all(stamped), "some message not stamped"

    hid = History(messages=[s, h, a, t]).save("ts_tester")
    loaded = History.load_from_id(hid)

    by_type = {type(m).__name__: m for m in loaded.messages}
    for orig, cls in [(s, "SystemMessage"), (h, "HumanMessage"), (a, "AIMessage"), (t, "ToolMessage")]:
        got = by_type[cls].additional_kwargs.get("timestamp")
        assert got == orig.additional_kwargs["timestamp"], f"{cls} timestamp lost on reload"


def test_fork_preserves_timestamps(tmp_path, monkeypatch):
    """SI forks histories — the fork must keep per-message timestamps (fork copies live objects)."""
    monkeypatch.setenv("HEAVEN_DATA_DIR", str(tmp_path))

    h = _stamp_ts(HumanMessage(content="do the thing"))
    a = _stamp_ts(AIMessage(content="done"))
    hid = History(messages=[h, a]).save("fork_tester")

    fork = fork_history(hid)                      # returns the saved fork History
    reloaded = History.load_from_id(fork.history_id)
    fh = next(m for m in reloaded.messages if isinstance(m, HumanMessage))
    fa = next(m for m in reloaded.messages if isinstance(m, AIMessage))
    assert fh.additional_kwargs.get("timestamp") == h.additional_kwargs["timestamp"], \
        "forked HUMAN timestamp lost"
    assert fa.additional_kwargs.get("timestamp") == a.additional_kwargs["timestamp"], \
        "forked AI timestamp lost"
