"""Regression: a standalone stdin-hook script (Claude Code contract) that
sys.exit()s at module level must NOT kill the host process when the devdir
walk exec_modules it — _load_devdir_hook converts SystemExit to ImportError so
the caller's except-Exception skip path handles it (proven live 2026-07-10:
any agent constructed from /home/GOD died at codenose_pretool.py's exit 0)."""

import pytest

from heaven_base.baseheavenagent import BaseHeavenAgent


def test_systemexit_hook_becomes_importerror(tmp_path):
    script = tmp_path / "alien_stdin_hook.py"
    script.write_text("import sys\nsys.exit(0)\n")
    with pytest.raises(ImportError, match="exited at import"):
        BaseHeavenAgent._load_devdir_hook(None, script)


def test_valid_module_still_loads(tmp_path):
    mod = tmp_path / "real_hook.py"
    mod.write_text("MARKER = 42\n")
    loaded = BaseHeavenAgent._load_devdir_hook(None, mod)
    assert loaded.MARKER == 42
