# ProgramBench samples

Eleven ProgramBench tasks run through NeMo Gym's Harbor bridge (`responses_api_agents/harbor_agent`).

Each task is a Harbor task package. The agent gets a program's documentation and a
test harness and has to write a program that behaves like the one the documentation
describes. The package's own verifier runs the hidden tests and writes the reward.
The reward is the fraction of hidden tests passed, from 0.0 to 1.0; `resolved` is
1.0 only when every test passes.

The packages are not part of this repository. They live in
`SanFranciscoData/NVIDIA-Training-Samples` under `programbench/tasks/<task>/`.
`manifest.json` pins each package by a content hash over every file in it, and
preparation refuses a package whose bytes differ from the pin.

## Prerequisites

- Linux with Docker (the packages build from their own Dockerfiles; about 10 GB of images in total).
- NeMo Gym set up as in the repository README (`uv venv && uv sync --extra dev`, Python 3.13.14).
- The Harbor agent's own environment: `gym env start` creates it from
  `responses_api_agents/harbor_agent/requirements.txt`. It must also contain `pathspec`,
  which Harbor imports and which the repository's uv overrides exclude:
  `uv pip install --python responses_api_agents/harbor_agent/.venv/bin/python pathspec`.
- A model endpoint that speaks the OpenAI chat API, named in `env.yaml` as
  `policy_base_url`, `policy_api_key` and `policy_model_name`.

## Run

```sh
# 1. The task packages, as a directory or a symlink named tasks/ in this directory.
git clone https://github.com/SanFranciscoData/NVIDIA-Training-Samples /tmp/nvidia-training-samples
ln -s /tmp/nvidia-training-samples/programbench/tasks benchmarks/programbench_samples/tasks

# 2. Verify the pins and write data/benchmark.jsonl.
gym eval prepare --benchmark programbench_samples

# 3. Start the servers and run one rollout per task.
gym env start --config benchmarks/programbench_samples/config.yaml --model-type openai_model
gym eval run --no-serve \
  --agent programbench_samples_harbor_agent \
  --input benchmarks/programbench_samples/data/benchmark.jsonl \
  --output results/programbench_samples/rollouts.jsonl
```

Overrides: `++programbench_tasks_dir=...`, `++programbench_jobs_dir=...`,
`++programbench_concurrency=N` (one Docker environment per task in flight),
`++programbench_max_turns=N`, and `++num_repeats=8` for pass@8.

## Network policy

The sandbox runs with Docker network mode `none`
(`no_network_env.py`, set by `harbor_environment_import_path` in `config.yaml`).
The agent and the verifier are driven from the host, and the model is called
from the agent server, so nothing inside the sandbox needs connectivity. With an
open network the agent can clone the program's upstream repository and ship the
original binary, which scores 1.0 without writing anything; that is why the
policy is enforced in the environment rather than documented.

The packages also declare Harbor's `network_mode = "allowlist"` in `task.toml`.
The Harbor version pinned by the bridge does not know that field and ignores it;
the environment class here is what enforces the policy under Gym.

## Equivalence with direct Harbor

For the same task and the same submission, direct `harbor run` (Harbor 0.23.0)
and this path give the same reward: an oracle submission scores 1.0 on both, an
empty submission 0.0 on both with the same error code, and a partial submission
0.16666666666666666 on both, with identical per-test outcomes and an identical
built binary. The files the agent can see in `/workspace` are the same on both
paths; the hidden tests are not visible on either.

One difference remains. The direct ProgramBench measurement uses mini-swe-agent
with the package's `cheat_instruction.txt` wrapped around `instruction.md`; this
path uses Harbor's Terminus-2 agent with `instruction.md` alone. Rewards for a
given submission are identical; pass rates for a model are not expected to be.

## Smoke rollouts

One rollout per task with `zai-org/GLM-5.3` through Gym's `openai_model` server,
`max_turns` 60, network `none`:

| task | reward | resolved |
|------|--------|----------|
| (filled in from the run) | | |
