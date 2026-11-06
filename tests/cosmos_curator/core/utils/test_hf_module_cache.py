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

"""Fresh-process checks for the worker's private Transformers module cache."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_cache_is_local_early_and_removed_on_exit(tmp_path: Path, exit_code: int) -> None:
    """Only the import cache moves; repeated setup and normal/error exits retain correct ownership."""
    shared = tmp_path / "shared"
    shared.mkdir()
    sentinel = shared / "weights.bin"
    sentinel.write_bytes(b"shared weights")
    code = """
import json
import os
import sys
from pathlib import Path
from cosmos_curator.core.utils.hf_module_cache import configure_hf_modules_cache
assert "transformers" not in sys.modules
configure_hf_modules_cache()
first = os.environ["HF_MODULES_CACHE"]
configure_hf_modules_cache()
assert os.environ["HF_MODULES_CACHE"] == first
assert "transformers" not in sys.modules
from transformers.utils import HF_MODULES_CACHE
assert HF_MODULES_CACHE == first
assert Path(first).is_dir()
print(json.dumps({"cache": first, "home": os.environ["HF_HOME"], "hub": os.environ["HF_HUB_CACHE"]}))
sys.exit(int(sys.argv[1]))
"""
    result = subprocess.run(  # noqa: S603 -- fixed Python script; no shell
        [sys.executable, "-c", code, str(exit_code)],
        env={
            **os.environ,
            "HF_MODULES_CACHE": str(shared),
            "HF_HOME": str(shared),
            "HF_HUB_CACHE": str(shared),
            "TMPDIR": str(shared),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == exit_code, result.stderr
    info = json.loads(result.stdout)
    assert Path(info["cache"]).parent == Path("/tmp")  # noqa: S108 -- node-local cache contract
    assert not Path(info["cache"]).exists()
    assert info["home"] == info["hub"] == str(shared)
    assert list(shared.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"shared weights"


def test_worker_cache_rejects_configuration_after_transformers_import() -> None:
    """A late hook must fail explicitly instead of leaving Transformers on a shared cache."""
    code = """
from transformers.utils import HF_MODULES_CACHE
from cosmos_curator.core.utils.hf_module_cache import configure_hf_modules_cache
try:
    configure_hf_modules_cache()
except RuntimeError as error:
    assert "before importing Transformers" in str(error)
else:
    raise AssertionError("Late configuration was silently accepted")
"""
    result = subprocess.run(  # noqa: S603 -- fixed script
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_launcher_preserves_script_arguments_imports_and_exit_code(tmp_path: Path, exit_code: int) -> None:
    """Ray's script must run normally with a private cache selected before its first import."""
    (tmp_path / "sibling.py").write_text("value = 42\n")
    script = tmp_path / "worker.py"
    script.write_text(
        "from transformers.utils import HF_MODULES_CACHE\n"
        "import sys\n"
        "import sibling\n"
        "assert sibling.value == 42\n"
        "assert sys.argv[2] == 'argument with spaces'\n"
        "print(HF_MODULES_CACHE)\n"
        "sys.exit(int(sys.argv[1]))\n"
    )
    result = subprocess.run(  # noqa: S603 -- fixed launcher and generated worker fixture
        [
            sys.executable,
            "-m",
            "cosmos_curator.core.utils.hf_module_cache",
            str(script),
            str(exit_code),
            "argument with spaces",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == exit_code, result.stderr
    cache = Path(result.stdout.strip())
    assert cache.parent == Path("/tmp")  # noqa: S108 -- node-local cache contract
    assert not cache.exists()


@pytest.mark.parametrize("site_flags", [("-s",), ("-S",), ("-S", "-s")])
def test_worker_launcher_applies_ray_forwarded_site_flags(tmp_path: Path, site_flags: tuple[str, ...]) -> None:
    """Ray places Python's site flags before the worker script when the driver disables site packages."""
    script = tmp_path / "worker.py"
    script.write_text(
        "import json, os, sys\n"
        "print(json.dumps({'cache': os.environ['HF_MODULES_CACHE'], 'args': sys.argv[1:], "
        "'no_site': sys.flags.no_site, 'no_user_site': sys.flags.no_user_site}))\n"
    )
    result = subprocess.run(  # noqa: S603 -- fixed launcher and generated worker fixture
        [
            sys.executable,
            "-m",
            "cosmos_curator.core.utils.hf_module_cache",
            *site_flags,
            str(script),
            "argument with spaces",
            "-s",
        ],
        env={name: value for name, value in os.environ.items() if name != "PYTHONNOUSERSITE"},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    info = json.loads(result.stdout)
    assert info["args"] == ["argument with spaces", "-s"]
    assert info["no_user_site"] == int("-s" in site_flags)
    assert info["no_site"] == int("-S" in site_flags)
    assert not Path(info["cache"]).exists()
