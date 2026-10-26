# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime storage access shared by checkpointed enrichment and append bindings."""

from cosmos_curator.core.utils.storage.storage_utils import get_lance_storage_options
from cosmos_curator.next.core.config import CheckpointedExecutionConfig
from cosmos_curator.next.utils.arrow_utils import s3_compatibility_env_vars


class RecoveryBindings:
    """Resolve storage credentials for a binding's table and staging root at runtime, never retaining them.

    ``EnrichmentBindings`` and ``AppendBindings`` mix this in; they provide the
    table ``uri``, the ``staging_root``, and ``execution`` settings naming the
    storage profile.
    """

    __slots__ = ()
    uri: str
    staging_root: str
    execution: CheckpointedExecutionConfig

    @property
    def storage_options(self) -> dict[str, str] | None:
        """Resolve the current runtime credentials without retaining them in bindings."""
        return get_lance_storage_options(self.uri, profile_name=self.execution.storage_profile)

    def storage_env_vars(self, *locations: str) -> dict[str, str]:
        """Return the Ray worker storage environment for the table, staging root, and the recipe's locations.

        Pass every other location the recipe's workers read or write, such as source media.
        """
        return s3_compatibility_env_vars(
            self.uri, self.staging_root, *locations, storage_profile=self.execution.storage_profile
        )
