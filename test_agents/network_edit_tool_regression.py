from pathlib import Path

import pytest

from heaven_base.tools.network_edit_tool import (
    EditHelper,
    NetworkEditToolArgsSchema,
    normalize_target_container,
)


def test_normalize_target_container_local_sentinels():
    for value in ("local", "LOCAL", " local "):
        assert normalize_target_container(value) is None


def test_normalize_target_container_preserves_real_container_name():
    assert normalize_target_container(None) is None
    assert normalize_target_container("") == ""
    assert normalize_target_container("host") == "host"
    assert normalize_target_container("localhost") == "localhost"
    assert normalize_target_container("mind_of_god") == "mind_of_god"


def test_target_container_schema_stays_required():
    schema = NetworkEditToolArgsSchema()
    target_container = schema.arguments["target_container"]
    assert target_container["required"] is True


@pytest.mark.asyncio
async def test_local_target_container_views_local_file_without_docker(tmp_path, monkeypatch):
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta\n")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("local target_container must not call docker subprocesses")

    monkeypatch.setattr("heaven_base.tools.network_edit_tool.subprocess.run", fail_if_called)

    result = await EditHelper().use_edit_helper(
        command="view",
        path=str(path),
        command_arguments={},
        target_container="local",
    )

    assert "alpha" in result.output
    assert "beta" in result.output


@pytest.mark.asyncio
async def test_local_target_container_creates_local_file(tmp_path, monkeypatch):
    path = tmp_path / "created.txt"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("local target_container must not call docker subprocesses")

    monkeypatch.setattr("heaven_base.tools.network_edit_tool.subprocess.run", fail_if_called)

    result = await EditHelper().use_edit_helper(
        command="create",
        path=str(path),
        command_arguments={"file_text": "created locally"},
        target_container="local",
    )

    assert "File created successfully" in result.output
    assert path.read_text() == "created locally"
