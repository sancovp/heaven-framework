"""Per-message happen-time stamping (_stamp_ts) survives a real History save -> load round-trip.

LangChain messages carry no native timestamp; heaven stamps it into additional_kwargs at
receipt/creation. History must round-trip additional_kwargs for BOTH Human and AI messages (AI already
did; Human's dict-load paths were dropping it). This is the datetime identity carton/HS keys off.
"""
import os

from langchain_core.messages import HumanMessage, AIMessage

from heaven_base.baseheavenagent import _stamp_ts
from heaven_base.memory.history import History


def test_stamp_ts_sets_and_never_overwrites():
    h = _stamp_ts(HumanMessage(content="hi"))
    assert h.additional_kwargs.get("timestamp"), "human message not stamped"
    first = h.additional_kwargs["timestamp"]
    _stamp_ts(h)  # setdefault — must NOT overwrite
    assert h.additional_kwargs["timestamp"] == first


def test_timestamp_survives_history_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAVEN_DATA_DIR", str(tmp_path))

    h = _stamp_ts(HumanMessage(content="what time is it"))
    a = _stamp_ts(AIMessage(content="now"))
    assert h.additional_kwargs.get("timestamp")
    assert a.additional_kwargs.get("timestamp")

    hist = History(messages=[h, a])
    hid = hist.save("ts_tester")

    loaded = History.load_from_id(hid)
    hm = next(m for m in loaded.messages if isinstance(m, HumanMessage))
    am = next(m for m in loaded.messages if isinstance(m, AIMessage))

    # both timestamps must survive save -> load (Human was the one being dropped before)
    assert hm.additional_kwargs.get("timestamp") == h.additional_kwargs["timestamp"], \
        "HUMAN message timestamp lost on reload"
    assert am.additional_kwargs.get("timestamp") == a.additional_kwargs["timestamp"], \
        "AI message timestamp lost on reload"
