# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the ProgramBench samples index from the pinned Harbor task packages.

Each task is a Harbor task directory (task.toml, instruction.md, environment/, tests/).
``manifest.json`` pins every package by a content hash; preparation refuses a package
whose bytes differ from the pin, so a rollout always runs the task that was measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BENCHMARK_DIR / "manifest.json"
OUTPUT_PATH = BENCHMARK_DIR / "data" / "benchmark.jsonl"
DEFAULT_TASKS_DIR = BENCHMARK_DIR / "tasks"
DATASET_ALIAS = "programbench"
AGENT_NAME = "programbench_samples_harbor_agent"


def content_hash(task_dir: Path) -> str:
    """Hash of every regular file under the package, by path and bytes, independent of mtimes and host."""
    digest = hashlib.sha256()
    for path in sorted(p for p in task_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(task_dir).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    manifest = json.loads(manifest_path.read_text())
    names = [task["name"] for task in manifest["tasks"]]
    if len(names) != len(set(names)):
        raise ValueError("manifest task names must be unique")
    return manifest


def prepare(
    *,
    tasks_dir: Path | None = None,
    manifest_path: Path = MANIFEST_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Verify every pinned package and write the benchmark rows."""
    tasks_dir = (tasks_dir or DEFAULT_TASKS_DIR).resolve()
    manifest = load_manifest(manifest_path)
    rows: list[str] = []
    for task in manifest["tasks"]:
        task_dir = tasks_dir / task["name"]
        if not (task_dir / "task.toml").is_file():
            raise FileNotFoundError(
                f"{task_dir} is not a Harbor task package; see README for where the tasks come from"
            )
        actual = content_hash(task_dir)
        if actual != task["ref"]:
            raise ValueError(f"{task['name']}: package hash {actual} does not match the pinned {task['ref']}")
        rows.append(
            json.dumps(
                {
                    "instance_id": f"{DATASET_ALIAS}::{task['name']}",
                    "task_id": task["name"],
                    "task_ref": task["ref"],
                    "responses_create_params": {"input": []},
                    "agent_ref": {"name": AGENT_NAME, "type": "responses_api_agents"},
                },
                sort_keys=True,
            )
            + "\n"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(rows), encoding="utf-8")
    print(f"Wrote {len(rows)} ProgramBench tasks to {output_path}", flush=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, default=None, help=f"task packages (default {DEFAULT_TASKS_DIR})")
    args = parser.parse_args()
    prepare(tasks_dir=args.tasks_dir)


if __name__ == "__main__":
    main()
