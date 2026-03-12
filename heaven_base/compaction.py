"""Compaction prompts and helpers for BaseHeavenAgent.

Pure data module — no imports from heaven_base to avoid circular dependencies.
BaseHeavenAgent imports from here; never the other way around.
"""

import re
from typing import List


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

COMPACTION_SYSTEM_PROMPT = (
    "You are in COMPACTION MODE. Your ONLY job is to produce an exhaustive, "
    "detailed, chronological narrative of everything that happened in this conversation. "
    "You have NO tools. Just read the history and produce summaries.\n\n"
    "CRITICAL RULES:\n"
    "- Write as if you are producing a meticulous incident log for someone who was NOT present.\n"
    "- Be EXHAUSTIVE. Include every file path, every command, every tool call, every error message.\n"
    "- Narrate chronologically: 'First, the user asked X. Then the agent did Y. The result was Z.'\n"
    "- Include EXACT file paths and container names.\n"
    "- Include the EXACT commands that were run and their outputs when relevant.\n"
    "- Include EXACT error messages when things failed, and what was done to fix them.\n"
    "- Include the reasoning and decisions made: WHY something was done, not just WHAT.\n"
    "- Include the user's stated goals, preferences, frustrations, and standing instructions.\n"
    "- Include any patterns, workarounds, or conventions that were discovered or established.\n"
    "- If something is in progress, describe EXACTLY what was being done and what the next step is.\n"
    "- Do NOT compress, abbreviate, or abstract. The next instance needs the FULL picture.\n"
    "- Do NOT editorialize or add commentary. Just describe what happened, faithfully and exactly.\n"
    "- Your summaries should be LONG. A thorough retelling of a long conversation should be long.\n\n"
    "Output your narrative inside <COMPACTION_SUMMARY> blocks. Use as many blocks as needed.\n"
    "If you need to do something after compaction (check a file, run a command, etc), "
    "put it in the summary as a concrete reminder with exact commands. "
    "Do NOT try to do it now."
)

COMPACTION_USER_PROMPT = (
    "# COMPACTION MODE — PRODUCE EXHAUSTIVE NARRATIVE\n\n"
    "Your context is about to be wiped. A NEW instance of you — with ZERO memory — "
    "will receive ONLY what you write here. If you leave something out, it is GONE FOREVER.\n\n"
    "## Instructions\n\n"
    "Go through the ENTIRE conversation history from beginning to end. For each phase "
    "of work, write a detailed <COMPACTION_SUMMARY> block that narrates EXACTLY what "
    "happened, as if telling the story to someone who was not present.\n\n"
    "### What to include in EVERY block:\n"
    "- What the user asked for and WHY (their stated goals and reasoning)\n"
    "- What actions were taken: exact commands, exact file paths, exact tool calls\n"
    "- What the results were: outputs, errors, successes\n"
    "- What decisions were made and WHY\n"
    "- What was learned: working patterns, container names, path conventions, gotchas\n"
    "- If something failed, the EXACT error and what was done about it\n\n"
    "### Format\n\n"
    "<COMPACTION_SUMMARY>\n"
    "[Detailed chronological narrative of this chunk of work. Be specific and thorough.\n"
    "Include file paths, command outputs, error messages, decisions, and reasoning.\n"
    "This should read like a detailed incident report, not a bullet-point summary.\n"
    "The next instance should be able to understand EXACTLY what happened and continue\n"
    "without asking 'wait, what was that about?']\n"
    "</COMPACTION_SUMMARY>\n\n"
    "### Rules\n"
    "- Be EXHAUSTIVE. Long is good. Thorough is good. Vague is UNACCEPTABLE.\n"
    "- Write in plain narrative prose, chronologically.\n"
    "- Include the user's current goals, standing instructions, preferences, and frustrations.\n"
    "- Include infrastructure knowledge: which containers exist, how to access them, "
    "what's installed where, which paths work.\n"
    "- If work is in progress, describe EXACTLY where it left off and what to do next.\n"
    "- Include environmental facts: environment variables, config file locations, "
    "service URLs, running processes.\n"
    "- Do NOT compress or abstract. The next instance needs the FULL picture.\n"
    "- Do NOT skip things because they seem minor. Minor details are often critical.\n"
    "- Output as many blocks as needed. A thorough retelling of a long conversation should be LONG.\n\n"
    "When you have covered EVERYTHING, output <COMPACTION_COMPLETE/> as the very last line."
)

COMPACTION_CONTINUE_PROMPT = (
    "Continue your compaction summary from where you left off. "
    "You have NOT finished yet. Keep going through the conversation history "
    "and produce more <COMPACTION_SUMMARY> blocks for the parts you haven't covered yet.\n\n"
    "When you have covered EVERYTHING, output <COMPACTION_COMPLETE/> as the very last line."
)

COMPACTION_BOOTSTRAP_TEMPLATE = (
    "# CONTINUATION AFTER COMPACTION\n\n"
    "Your conversation history was just compacted. The summary below contains "
    "everything from the previous conversation. Use it to orient yourself, "
    "then continue working.\n\n"
    "<COMPACTION_SUMMARY>\n{summary}\n</COMPACTION_SUMMARY>"
)

DEFAULT_COMPACT_THRESHOLD = 800_000
DEFAULT_MAX_COMPACTION_PASSES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUMMARY_PATTERN = re.compile(
    r'<COMPACTION_SUMMARY>(.*?)</COMPACTION_SUMMARY>', re.DOTALL
)


def parse_compaction_summaries(text: str) -> List[str]:
    """Extract all <COMPACTION_SUMMARY> blocks from text."""
    return [block.strip() for block in _SUMMARY_PATTERN.findall(text)]
