"""Claude-Code plugin loader for heaven — install (most) CC plugins UNCHANGED.

A Claude-Code plugin = a `.claude-plugin/plugin.json` manifest + component directories. heaven already loads
each component type, so "installing" a CC plugin is just: read the manifest, then route each component into
the target `.heaven` directory, where heaven's existing loaders pick it up:

  skills/   -> <heaven>/skills     (auto-loaded by hooks.skill_autoload — name+desc+path injected)
  agents/   -> <heaven>/agents      (resolved by use_hermes / the agent loaders)
  rules/    -> <heaven>/rules        (auto-loaded by BaseHeavenAgent.resolve_devdirs — the native devdir loader)
  hooks/    -> <heaven>/hooks
  commands/ -> <heaven>/commands     (the slash-command shim consumes these — TODO)
  .mcp.json -> merged into <heaven>/mcp.json   (mcpServers)

Dep-free (stdlib only — NO skill_manager/carton). Records every install in `<heaven>/plugins.json`.
This is the heaven side of Claude-Code-plugin parity (DESIGN §12.3).
"""
import json
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Component dirs copied verbatim into <heaven>/<comp>. `.mcp.json` is merged separately.
COMPONENT_DIRS = ("skills", "agents", "rules", "hooks", "commands")


def read_plugin_manifest(plugin_dir) -> Optional[Dict[str, Any]]:
    """Read a CC plugin manifest (`.claude-plugin/plugin.json`, or `plugin.json` at the root as a fallback)."""
    base = Path(plugin_dir)
    for cand in (base / ".claude-plugin" / "plugin.json", base / "plugin.json"):
        if cand.is_file():
            try:
                return json.loads(cand.read_text())
            except Exception as e:
                logger.warning("plugin manifest unreadable (%s): %s", cand, e)
                return None
    return None


def _merge_mcp(plugin_dir: Path, heaven_dir: Path) -> int:
    """Merge a plugin's `.mcp.json` mcpServers into `<heaven>/mcp.json`. Returns the count merged."""
    src = plugin_dir / ".mcp.json"
    if not src.is_file():
        return 0
    try:
        incoming = (json.loads(src.read_text()) or {}).get("mcpServers", {}) or {}
    except Exception:
        return 0
    if not incoming:
        return 0
    dst = heaven_dir / "mcp.json"
    try:
        cur = json.loads(dst.read_text()) if dst.is_file() else {}
    except Exception:
        cur = {}
    servers = cur.get("mcpServers", {}) if isinstance(cur, dict) else {}
    servers.update(incoming)
    dst.write_text(json.dumps({"mcpServers": servers}, indent=2) + "\n")
    return len(incoming)


def install_cc_plugin(plugin_dir, heaven_dir=None) -> Dict[str, Any]:
    """Install a Claude-Code plugin into a `.heaven` dir (default `~/.heaven`).

    Reads `.claude-plugin/plugin.json`, copies each component dir into `<heaven>/<comp>`, merges any
    `.mcp.json`, and records the install in `<heaven>/plugins.json`. Returns a summary
    `{name, version, heaven_dir, installed: {component: count}}`. Idempotent (dirs merged, files overwritten).
    """
    plugin_dir = Path(plugin_dir).expanduser().resolve()
    if not plugin_dir.is_dir():
        raise FileNotFoundError(f"plugin dir not found: {plugin_dir}")
    heaven_dir = Path(heaven_dir).expanduser() if heaven_dir else (Path.home() / ".heaven")
    heaven_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_plugin_manifest(plugin_dir) or {}
    name = manifest.get("name") or plugin_dir.name

    installed: Dict[str, int] = {}
    for comp in COMPONENT_DIRS:
        src = plugin_dir / comp
        if not src.is_dir():
            continue
        dst = heaven_dir / comp
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for item in sorted(src.iterdir()):
            if item.name.startswith("."):
                continue
            target = dst / item.name
            try:
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
                n += 1
            except Exception as e:
                logger.warning("plugin %s: failed to install %s/%s: %s", name, comp, item.name, e)
        if n:
            installed[comp] = n

    mcp_n = _merge_mcp(plugin_dir, heaven_dir)
    if mcp_n:
        installed["mcpServers"] = mcp_n

    reg = heaven_dir / "plugins.json"
    try:
        data = json.loads(reg.read_text()) if reg.is_file() else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    data[name] = {"source": str(plugin_dir), "version": manifest.get("version"), "installed": installed}
    reg.write_text(json.dumps(data, indent=2) + "\n")

    logger.info("installed CC plugin '%s' into %s: %s", name, heaven_dir, installed)
    return {"name": name, "version": manifest.get("version"), "heaven_dir": str(heaven_dir), "installed": installed}


def list_installed_plugins(heaven_dir=None) -> Dict[str, Any]:
    """Return the `<heaven>/plugins.json` registry of installed plugins."""
    heaven_dir = Path(heaven_dir).expanduser() if heaven_dir else (Path.home() / ".heaven")
    reg = heaven_dir / "plugins.json"
    try:
        return json.loads(reg.read_text()) if reg.is_file() else {}
    except Exception:
        return {}
