# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor Docker environment whose sandbox has no network.

The agent and the verifier are driven from the host through ``docker exec``; the
model is called from the agent server, never from inside the sandbox. Nothing in
the container needs connectivity, so the container gets none.
"""

from __future__ import annotations

from harbor.environments.docker.docker import DockerEnvironment


class NoNetworkDockerEnvironment(DockerEnvironment):
    """``DockerEnvironment`` with the compose network mode forced to ``none``."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._env_vars.network_mode = "none"
