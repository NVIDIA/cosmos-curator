#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Secret extractor script for init container.

Extracts certificate and key from NVCF secrets JSON and writes them to the appropriate location.
"""

import json
import logging
import time
from pathlib import Path
from typing import NoReturn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("secret_extractor")

SECRETS_FILE = Path("/var/secrets/secrets.json")
OUTPUT_DIR = Path("/etc/curator-remote-write/certs")
CERT_FILE = OUTPUT_DIR / "tls.crt"
KEY_FILE = OUTPUT_DIR / "tls.key"

# Secret keys in JSON
CERT_KEY = "byo-metrics-receiver-client-crt"
KEY_KEY = "byo-metrics-receiver-client-key"

# Timing constants
MAX_WAIT_TIME = 300  # 5 minutes in seconds
WAIT_INTERVAL = 5  # Check every 5 seconds


def wait_for_secrets_file() -> None:
    """Wait for secrets file to appear, up to MAX_WAIT_TIME seconds."""
    deadline = time.time() + MAX_WAIT_TIME

    while time.time() < deadline:
        if SECRETS_FILE.exists():
            return
        logger.info("Waiting for secrets file %s to appear...", SECRETS_FILE)
        time.sleep(WAIT_INTERVAL)

    error_msg = f"Secrets file {SECRETS_FILE} did not appear within {MAX_WAIT_TIME} seconds"
    logger.error(error_msg)
    raise TimeoutError(error_msg)


def read_secrets() -> tuple[str, str]:
    """Read and validate secrets from JSON file."""
    try:
        secrets = json.loads(SECRETS_FILE.read_text())
    except json.JSONDecodeError as err:
        # Don't include decode error as it might contain parts of the secret
        error_msg = f"Failed to parse {SECRETS_FILE} as JSON"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from err
    except OSError as err:
        error_msg = f"Failed to read {SECRETS_FILE}: {err}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from err

    missing = [key for key in (CERT_KEY, KEY_KEY) if not secrets.get(key)]
    if missing:
        error_msg = f"Missing or empty required secrets: {', '.join(missing)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    return secrets[CERT_KEY], secrets[KEY_KEY]


def write_secrets(cert: str, key: str) -> None:
    """Write certificate and key to output directory."""
    try:
        CERT_FILE.write_text(cert)
        KEY_FILE.write_text(key)
    except OSError as err:
        error_msg = f"Failed to write secrets to {OUTPUT_DIR}: {err}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from err


def main() -> NoReturn:
    """Extract secrets from NVCF secrets JSON and write them to the appropriate location."""
    logger.info("Starting secret extraction process...")
    wait_for_secrets_file()
    logger.info("Found secrets file %s", SECRETS_FILE)

    cert, key = read_secrets()
    logger.info("Successfully read secrets")

    write_secrets(cert, key)
    logger.info("Successfully wrote secrets to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
