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
"""Module: Utilities for Summarizing Imaginaire Datasets.

This module provides utilities to track and summarize progress on datasets, particularly useful for
monitoring completion percentages on a total, per-root, and per-aspect-ratio-directory basis.
"""

import contextlib
import pathlib
from collections.abc import Generator, Iterable
from typing import Any

import attrs
import numpy as np
import pandas as pd
from loguru import logger


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary into a single-level dictionary.

    Args:
        d: Input dictionary to flatten.
        parent_key: Key prefix for nested dictionaries.
        sep: Separator to use between nested keys.

    Returns:
        Flattened dictionary with concatenated keys.

    """
    items: list[Any] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def attrs_to_df(objects: Iterable[Any], *, flatten: bool = False) -> pd.DataFrame:
    """Objects are classes derived from attrs."""
    dicts = [flatten_dict(attrs.asdict(obj)) for obj in objects] if flatten else [attrs.asdict(obj) for obj in objects]
    return pd.DataFrame(dicts)


@contextlib.contextmanager
def turn_off_pandas_display_limits() -> Generator[None, None, None]:
    """Temporarily disable pandas display limits.

    This context manager turns off limitations on the number of displayed rows and columns,
    as well as the max column width and display width in pandas DataFrames. This is useful
    when logging or printing large DataFrames for comprehensive visibility.

    Yields:
        Generator[None, None, None]: A generator yielding None, used to manage context.

    """
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.max_colwidth",
        None,
    ):
        yield


@contextlib.contextmanager
def turn_off_numpy_display_limits() -> Generator[None, None, None]:
    """Context manager to temporarily set NumPy print options to print the full array."""
    original_options = np.get_printoptions()
    npmax = np.iinfo(np.int32).max
    np.set_printoptions(threshold=npmax, linewidth=npmax)
    try:
        yield
    finally:
        np.set_printoptions(**original_options)


@attrs.define(order=True)
class SummaryForAspectRatioDir:
    """Represents a summarized data entity for a specific aspect ratio directory.

    Attributes:
        source_dir (pathlib.Path): The source directory path.
        dest_dir (pathlib.Path): The destination directory path.
        root_dir (pathlib.Path): The root directory path.
        num_inputs (int): The number of input files/items.
        num_outputs (int): The number of output files/items.

    This class provides a structured way to encapsulate and compute summary information,
    such as the number of items left to process and the percentage completion.

    """

    source_dir: pathlib.Path
    dest_dir: pathlib.Path
    root_dir: pathlib.Path
    num_inputs: int
    num_outputs: int

    @property
    def num_left(self) -> int:
        """Calculates the number of items left to process."""
        return self.num_inputs - self.num_outputs

    @property
    def percentage_complete(self) -> float:
        """Calculates the percentage of work completed."""
        if self.num_inputs == 0:
            return 0.0
        return 100.0 * self.num_outputs / self.num_inputs

    def to_dict(self) -> dict[str, Any]:
        """Convert the summary data to a dictionary format.

        Returns:
            A dictionary containing the summary data.

        """
        summary_dict = attrs.asdict(self)
        summary_dict["num_left"] = self.num_left
        summary_dict["percentage_complete"] = self.percentage_complete
        return summary_dict


@attrs.define
class _SummarizedResults:
    """Encapsulates summarized results in various DataFrame formats.

    Attributes:
        per_aspect_ratio (pd.DataFrame): Summary DataFrame per aspect ratio.
        per_root (pd.DataFrame): Summary DataFrame per root directory.
        total (pd.Series): Total aggregated statistics.

    This class facilitates the organization and writing of summarized results to disk,
    providing an easy way to handle and store the computed summaries.

    """

    per_aspect_ratio: pd.DataFrame
    per_root: pd.DataFrame
    total: pd.DataFrame

    def write_to_disk(self, prefix: pathlib.Path) -> list[pathlib.Path]:
        """Write the summarized results to disk as CSV files.

        Args:
            prefix (pathlib.Path): The file path prefix for the CSV files.

        Returns:
            list[pathlib.Path]: A list of paths where the CSV files are saved.

        """
        paths = [
            pathlib.Path(f"{prefix}_per_aspect_ratio_dir.csv"),
            pathlib.Path(f"{prefix}_per_root_dir.csv"),
            pathlib.Path(f"{prefix}_total.csv"),
        ]
        self.per_aspect_ratio.to_csv(paths[0])
        self.per_root.to_csv(paths[1])
        self.total.to_csv(paths[2])
        return paths


def summarize_results_and_save(
    results: Iterable[SummaryForAspectRatioDir],
    prefix: pathlib.Path,
    *,
    log_total: bool = True,
) -> list[pathlib.Path]:
    """Summarizes dataset results and saves them to disk.

    The function processes a dataset structured in the format of root_dir/data_type_dir/aspect_ratio_dir/part_dir
    and summarizes the data on a per-root and per-aspect-ratio basis. It then writes these summaries to disk.

    Args:
        results (Iterable[SummaryForAspectRatioDir]): An iterable of SummaryForAspectRatioDir objects to summarize.
        prefix (pathlib.Path): The file path prefix for the summary CSV files.
        log_total (bool, optional): If True, logs the total summary. Defaults to True.

    Returns:
        list[pathlib.Path]: A list of file paths where the summary CSV files are saved.

    """
    per_aspect_ratio = pd.DataFrame([x.to_dict() for x in sorted(results)])

    per_root_df = per_aspect_ratio.groupby("root_dir")[["num_inputs", "num_outputs"]].sum()
    per_root_df["num_left"] = per_root_df["num_inputs"] - per_root_df["num_outputs"]
    per_root_df["percentage_complete"] = 100.0 * per_root_df["num_outputs"] / per_root_df["num_inputs"]

    total = per_aspect_ratio[["num_inputs", "num_outputs"]].sum()
    total["num_left"] = total["num_inputs"] - total["num_outputs"]
    total["percentage_complete"] = 100.0 * total["num_outputs"] / total["num_inputs"]

    summarized_results = _SummarizedResults(per_aspect_ratio, per_root_df, total.to_frame().T)
    if log_total:
        with turn_off_pandas_display_limits():
            logger.info(summarized_results.total)
    return summarized_results.write_to_disk(prefix)
