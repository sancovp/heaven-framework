"""Tests for heaven_base.memory.chat_branching (fork / template / rehydrate / compose).

No model calls — pure History/file mechanics against a tmp HEAVEN_DATA_DIR.
"""

import json
import os

import pytest

# EnvConfigUtil.get_env_value calls _update_env_val(), which re-sources
# ~/system_config.sh into os.environ on EVERY call — that would stomp the
# monkeypatched HEAVEN_DATA_DIR with the user's real one and write test
# artifacts into live data. No-op it so tests stay hermetic.


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    from heaven_base.utils.get_env_value import EnvConfigUtil

    monkeypatch.setattr(EnvConfigUtil, "_update_env_val", staticmethod(lambda: None))


@pytest.fixture()
def heaven_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HEAVEN_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def source_history(heaven_data_dir):
    """A saved 6-message history for agent test_agent: system, user, ai,
    ai(tool_calls), tool, ai."""
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    from heaven_base.memory.history import History

    h = History(messages=[
        SystemMessage(content="You are a test agent."),
        HumanMessage(content="Hello, my project is {{project}}."),
        AIMessage(content="Hi! Tell me about {{project}}."),
        AIMessage(content="", tool_calls=[
            {"name": "lookup", "args": {"q": "x"}, "id": "call_1"},
        ]),
        ToolMessage(content="lookup result", tool_call_id="call_1"),
        AIMessage(content="Based on the lookup, here is the answer."),
    ])
    h.save("test_agent")
    return h


@pytest.fixture()
def source_conversation(source_history):
    from heaven_base.memory.conversations import start_chat

    conv = start_chat("Test convo", source_history.history_id, "test_agent",
                      tags=["testing"])
    return conv


# ---------------------------------------------------------------- fork_history

def test_fork_full_copy_leaves_source_untouched(source_history):
    from heaven_base.memory.chat_branching import fork_history
    from heaven_base.memory.history import History

    fork = fork_history(source_history.history_id)

    assert fork.history_id != source_history.history_id
    assert "_fork_" in fork.history_id
    assert len(fork.messages) == 6
    assert fork.metadata["forked_from"] == source_history.history_id
    assert fork.metadata["fork_point"] == 6

    # source file still loads with original content
    src = History._load_history_file(source_history.history_id)
    assert len(src.messages) == 6

    # fork is loadable through the canonical continuation path
    cont = History.load_from_id(fork.history_id)
    assert len(cont.messages) == 6
    assert cont.history_id == f"{fork.history_id}_continued_1"


def test_fork_upto_slices_prefix(source_history):
    from heaven_base.memory.chat_branching import fork_history

    fork = fork_history(source_history.history_id, upto=3)
    assert len(fork.messages) == 3
    assert fork.metadata["fork_point"] == 3


def test_fork_trims_dangling_tool_call(source_history):
    from heaven_base.memory.chat_branching import fork_history

    # cutting at 4 would end on the AIMessage with tool_calls (index 3) whose
    # ToolMessage (index 4) is excluded -> must trim back to 3
    fork = fork_history(source_history.history_id, upto=4)
    assert len(fork.messages) == 3
    assert type(fork.messages[-1]).__name__ == "AIMessage"
    assert not getattr(fork.messages[-1], "tool_calls", None)


def test_fork_drops_agent_status_by_default(source_history):
    from heaven_base.memory.chat_branching import fork_history
    from heaven_base.memory.history import AgentStatus, History

    src = History._load_history_file(source_history.history_id)
    src.agent_status = AgentStatus(goal="g", task_list=["t"])
    src.save("test_agent")

    fork = fork_history(source_history.history_id)
    assert fork.agent_status is None

    kept = fork_history(source_history.history_id, keep_agent_status=True)
    assert kept.agent_status is not None and kept.agent_status.goal == "g"


# ---------------------------------------------------------- fork_conversation

def test_fork_conversation_lineage(source_conversation, source_history):
    from heaven_base.memory.chat_branching import fork_conversation
    from heaven_base.memory.conversations import ConversationManager

    result = fork_conversation(source_conversation["conversation_id"], upto=3,
                               title="My fork")

    assert result["title"] == "My fork"
    assert result["fork_point"] == 3

    new_conv = ConversationManager.load_conversation(result["conversation_id"])
    assert new_conv["history_chain"] == [result["history_id"]]
    meta = new_conv["metadata"]
    assert meta["forked_from_conversation"] == source_conversation["conversation_id"]
    assert meta["forked_from_history"] == source_history.history_id
    assert meta["fork_point"] == 3

    # source conversation untouched
    src = ConversationManager.load_conversation(source_conversation["conversation_id"])
    assert src["history_chain"] == [source_history.history_id]


# ---------------------------------------------------------------- ChatTemplate

def test_template_from_history_drops_system(source_history):
    from heaven_base.memory.chat_branching import ChatTemplate

    tmpl = ChatTemplate.from_history(source_history.history_id, name="t1")
    assert tmpl.messages[0]["type"] == "HumanMessage"
    assert len(tmpl.messages) == 5

    with_sys = ChatTemplate.from_history(source_history.history_id, name="t2",
                                         include_system=True)
    assert with_sys.messages[0]["type"] == "SystemMessage"


def test_template_store_roundtrip(source_history, heaven_data_dir):
    from heaven_base.memory.chat_branching import (
        ChatTemplate,
        delete_chat_template,
        get_chat_template,
        list_chat_templates,
        search_chat_templates,
    )

    tmpl = ChatTemplate.from_history(source_history.history_id, name="onboarding",
                                     description="Project onboarding context",
                                     tags=["onboard"])
    tid = tmpl.save()
    assert tid.startswith("tmpl_")
    assert (heaven_data_dir / "chat_templates" / f"{tid}.json").exists()

    by_id = get_chat_template(tid)
    by_name = get_chat_template("onboarding")
    assert by_id.template_id == by_name.template_id == tid
    assert by_id.messages == tmpl.messages

    listed = list_chat_templates()
    assert [t["template_id"] for t in listed] == [tid]
    assert listed[0]["message_count"] == 5

    assert search_chat_templates("onboard")[0]["template_id"] == tid
    assert search_chat_templates("zzz") == []

    assert delete_chat_template("onboarding") is True
    assert get_chat_template(tid) is None


def test_template_variable_substitution(source_history):
    from heaven_base.memory.chat_branching import ChatTemplate

    tmpl = ChatTemplate.from_history(source_history.history_id, name="v",
                                     variables={"project": "DEFAULT"})
    rendered = tmpl.render_messages()
    assert "my project is DEFAULT" in rendered[0]["content"]

    rendered = tmpl.render_messages({"project": "WOOM"})
    assert "my project is WOOM" in rendered[0]["content"]
    # literal braces elsewhere are untouched
    assert "{{" not in json.dumps(rendered)


def test_template_compose(source_history):
    from heaven_base.memory.chat_branching import ChatTemplate

    a = ChatTemplate.from_history(source_history.history_id, name="a", upto=3)
    b = ChatTemplate.from_history(source_history.history_id, name="b")
    c = a.compose(b, name="a+b")
    assert len(c.messages) == len(a.messages) + len(b.messages)
    assert c.source["composed_from"] == ["a", "b"]


def test_compose_chat_templates_store(source_history):
    from heaven_base.memory.chat_branching import (
        ChatTemplate,
        compose_chat_templates,
        get_chat_template,
    )

    ChatTemplate.from_history(source_history.history_id, name="x", upto=3).save()
    ChatTemplate.from_history(source_history.history_id, name="y").save()
    composed = compose_chat_templates(["x", "y"], name="xy")
    assert get_chat_template("xy").template_id == composed.template_id


# ------------------------------------------------------------------ rehydrate

def test_rehydrate_roundtrip(source_history):
    from heaven_base.memory.chat_branching import ChatTemplate, rehydrate_chat
    from heaven_base.memory.conversations import ConversationManager
    from heaven_base.memory.history import History

    tmpl = ChatTemplate.from_history(source_history.history_id, name="reh",
                                     variables={"project": "GAS"})
    tmpl.save()

    result = rehydrate_chat("reh", agent_name="other_agent", title="Rehydrated")
    assert result["message_count"] == 5
    assert "_rehydrated_" in result["history_id"]

    # live history loads through the canonical continuation path
    cont = History.load_from_id(result["history_id"])
    assert len(cont.messages) == 5
    assert "GAS" in cont.messages[0].content

    conv = ConversationManager.load_conversation(result["conversation_id"])
    assert conv["metadata"]["rehydrated_from_template"] == tmpl.template_id
    assert conv["history_chain"] == [result["history_id"]]


def test_rehydrate_missing_template_raises(heaven_data_dir):
    from heaven_base.memory.chat_branching import rehydrate_chat

    with pytest.raises(FileNotFoundError):
        rehydrate_chat("nope", agent_name="a")


# -------------------------------------------------------------- compose seams

def test_to_parsed_messages_shape(source_history):
    from heaven_base.memory.chat_branching import ChatTemplate

    tmpl = ChatTemplate.from_history(source_history.history_id, name="p",
                                     include_system=True)
    parsed = tmpl.to_parsed_messages({"project": "X"})
    assert parsed[0] == {"index": 0, "role": "system",
                         "content": "You are a test agent."}
    roles = [p["role"] for p in parsed]
    assert roles == ["system", "user", "assistant", "assistant", "tool", "assistant"]
    assert all(isinstance(p["content"], str) for p in parsed)


def test_to_context_dict_shape(source_history):
    from heaven_base.memory.chat_branching import ChatTemplate

    tmpl = ChatTemplate.from_history(source_history.history_id, name="ctx",
                                     description="d")
    ctx = tmpl.to_context_dict({"project": "X"})
    assert ctx["chat_template"] == "ctx"
    assert ctx["description"] == "d"
    assert isinstance(ctx["conversation"], list)
    assert ctx["conversation"][0].startswith("user: ")
