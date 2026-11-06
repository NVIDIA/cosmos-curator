# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Isolate Transformers' mutable Python-module cache before worker model imports.

Model weights and their staged Python sources may live on NFS/Lustre. Transformers
copies custom modules from those sources into HF_MODULES_CACHE using non-atomic
writes. Give each worker a private directory on /tmp so concurrent workers cannot
import one another's partially copied files. HF_HOME and weight caches are untouched.
"""

import atexit
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

_cache_owner_pid: int | None = None
_cache_directory: str | None = None


def _remove_owned_cache(directory: str, owner_pid: int) -> None:
    """Remove only this process's directory; forked children must not delete their parent's cache."""
    if os.getpid() == owner_pid:
        shutil.rmtree(directory, ignore_errors=True)


def configure_hf_modules_cache() -> None:
    """Select one private import cache for this worker's lifetime.

    Run before importing Transformers, whose module-cache constants are captured at
    import time. Normal interpreter exit removes the directory; SIGKILL/os._exit may
    leave small directories for the node's temporary-directory cleanup. Inherited
    HF_MODULES_CACHE and TMPDIR values are deliberately ignored: either can point at
    shared storage. This changes no model weight or download-cache settings.
    """
    global _cache_owner_pid, _cache_directory  # noqa: PLW0603 -- process-owned startup state
    pid = os.getpid()
    if _cache_owner_pid == pid and _cache_directory is not None:
        os.environ["HF_MODULES_CACHE"] = _cache_directory
        return
    if "transformers.utils.hub" in sys.modules or "transformers.dynamic_module_utils" in sys.modules:
        msg = "Configure HF_MODULES_CACHE before importing Transformers; worker initialization ran too late"
        raise RuntimeError(msg)
    directory = tempfile.mkdtemp(prefix="curator-hf-modules-", dir="/tmp")
    os.environ["HF_MODULES_CACHE"] = directory
    _cache_owner_pid = pid
    _cache_directory = directory
    atexit.register(_remove_owned_cache, directory, pid)


def main() -> None:
    """Run Ray's worker script in this interpreter after isolating its module cache."""
    arguments = sys.argv[1:]
    site_flags = []
    # Ray forwards these interpreter flags before the script. They must run before
    # Python imports site packages, rather than becoming arguments to this module.
    while arguments and arguments[0] in {"-s", "-S"}:
        site_flags.append(arguments.pop(0))
    if site_flags:
        os.execv(  # noqa: S606 -- current interpreter, fixed site flags, and this module's absolute path; no shell
            sys.executable,
            [sys.executable, *site_flags, str(Path(__file__).resolve()), *arguments],
        )
    configure_hf_modules_cache()
    sys.argv = arguments
    # Match Python's script execution semantics, including imports relative to the script directory.
    sys.path[0] = str(Path(sys.argv[0]).resolve().parent)
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
