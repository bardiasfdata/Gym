# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ProgramBench samples preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.programbench_samples import prepare as benchmark_prepare


def _write_package(tasks_dir: Path, name: str, body: str = "echo hi\n") -> Path:
    task_dir = tasks_dir / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text('[task]\nname = "%s"\n' % name)
    (task_dir / "instruction.md").write_text("Build the program.\n")
    (task_dir / "tests" / "test.sh").write_text(body)
    return task_dir


def _write_manifest(path: Path, tasks_dir: Path, names: list[str]) -> None:
    path.write_text(
        json.dumps(
            {"tasks": [{"name": n, "ref": benchmark_prepare.content_hash(tasks_dir / n)} for n in names]},
        )
    )


def test_content_hash_depends_on_bytes_and_paths_only(tmp_path: Path) -> None:
    a = _write_package(tmp_path / "a", "t")
    b = _write_package(tmp_path / "b", "t")
    assert benchmark_prepare.content_hash(a) == benchmark_prepare.content_hash(b)
    assert benchmark_prepare.content_hash(a).startswith("sha256:")
    (b / "tests" / "test.sh").write_text("echo changed\n")
    assert benchmark_prepare.content_hash(a) != benchmark_prepare.content_hash(b)
    c = _write_package(tmp_path / "c", "t")
    (c / "tests" / "test.sh").rename(c / "tests" / "other.sh")
    assert benchmark_prepare.content_hash(a) != benchmark_prepare.content_hash(c)


def test_prepare_writes_one_row_per_pinned_task(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    names = ["org--one.1234567", "org--two.89abcde"]
    for name in names:
        _write_package(tasks_dir, name)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, tasks_dir, names)
    output = tmp_path / "data" / "benchmark.jsonl"

    written = benchmark_prepare.prepare(tasks_dir=tasks_dir, manifest_path=manifest, output_path=output)

    rows = [json.loads(line) for line in written.read_text().splitlines()]
    assert [row["task_id"] for row in rows] == names
    for row, name in zip(rows, names):
        assert row["instance_id"] == f"programbench::{name}"
        assert row["task_ref"] == benchmark_prepare.content_hash(tasks_dir / name)
        assert row["responses_create_params"] == {"input": []}
        assert row["agent_ref"] == {"name": benchmark_prepare.AGENT_NAME, "type": "responses_api_agents"}


def test_prepare_refuses_a_package_that_differs_from_its_pin(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_package(tasks_dir, "org--one.1234567")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, tasks_dir, ["org--one.1234567"])
    (tasks_dir / "org--one.1234567" / "instruction.md").write_text("Build a different program.\n")

    with pytest.raises(ValueError, match="does not match the pinned"):
        benchmark_prepare.prepare(tasks_dir=tasks_dir, manifest_path=manifest, output_path=tmp_path / "out.jsonl")


def test_prepare_refuses_a_missing_package(tmp_path: Path) -> None:
    tasks_dir = tmp_path / "tasks"
    _write_package(tasks_dir, "org--one.1234567")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, tasks_dir, ["org--one.1234567"])

    with pytest.raises(FileNotFoundError):
        benchmark_prepare.prepare(tasks_dir=tmp_path / "elsewhere", manifest_path=manifest, output_path=tmp_path / "o")


def test_committed_manifest_pins_unique_tasks() -> None:
    manifest = benchmark_prepare.load_manifest()
    assert len(manifest["tasks"]) == 11
    assert all(task["ref"].startswith("sha256:") for task in manifest["tasks"])
