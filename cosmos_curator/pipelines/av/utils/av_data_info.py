# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Data information for AV pipelines."""

from typing import NotRequired, TypedDict


class CameraIdExtractor(TypedDict):
    """Configuration for extracting camera IDs from filenames.

    Attributes:
        delimiter: The delimiter used to split the filename.
        index: The index of the camera ID after splitting.

    """

    delimiter: str
    index: int


class CameraMapping(TypedDict):
    """Configuration for camera mapping and processing.

    Attributes:
        camera_id_extractor: Configuration for extracting camera IDs.
        camera_id_mapping_cosmos: Mapping of camera IDs to cosmos indices.
        camera_name_mapping_cosmos: Mapping of camera IDs to camera names.
        all_timestamp_files: List of timestamp file names.
        camera_id_for_vri_caption: List of camera IDs used for VRI captioning.
        align_every_frame: Whether to align every frame.
        extract_timestamp_from_video: Whether to extract timestamps from video.
        timestamp_camera_id_mapping: Mapping of timestamp IDs to camera IDs.
        flip_caption_input: Mapping of camera IDs to flip flags.

    """

    camera_id_extractor: CameraIdExtractor
    camera_id_mapping_cosmos: dict[int, int]
    camera_name_mapping_cosmos: dict[int, str]
    all_timestamp_files: list[str]
    camera_id_for_vri_caption: NotRequired[list[int]]
    align_every_frame: NotRequired[bool]
    extract_timestamp_from_video: NotRequired[bool]
    timestamp_camera_id_mapping: NotRequired[dict[int, int]]
    flip_caption_input: NotRequired[dict[int, bool]]


def get_avail_camera_format_ids() -> list[str]:
    """Get the available camera format IDs.

    Returns:
        The available camera format IDs.

    """
    return list(CAMERA_MAPPING.keys())


def get_camera_id(filename: str, delimiter: str, index: int) -> int:
    """Get the camera ID from the filename.

    Args:
        filename: The filename.
        delimiter: The delimiter.
        index: The index.

    Returns:
        The camera ID.

    """
    # extract camera id from filename like `cam-07-VIA-IMX390.h264`
    return int(filename.split(delimiter)[index])


CAMERA_MAPPING: dict[str, CameraMapping] = {
    "U": {
        # hard requirement
        "camera_id_extractor": {
            "delimiter": "-",
            "index": 1,
        },
        "camera_id_mapping_cosmos": {
            2: 0,
            5: 1,
            7: 2,
            6: 3,
            4: 4,
            8: 5,
        },
        "camera_name_mapping_cosmos": {
            2: "camera_front_wide_120fov",
            5: "camera_cross_left_120fov",
            7: "camera_cross_right_120fov",
            6: "camera_rear_tele_30fov",
            4: "camera_rear_left_70fov",
            8: "camera_rear_right_70fov",
        },
        "all_timestamp_files": [
            "timestamp_l1.csv",
            "timestamp_l2.csv",
            "timestamp_small.csv",
        ],
        "camera_id_for_vri_caption": [2],
        # soft requirement
        "align_every_frame": True,
        "extract_timestamp_from_video": False,
        "timestamp_camera_id_mapping": {
            2: 2,
            1: 3,
            7: 4,
            4: 5,
            0: 6,
            5: 7,
            6: 8,
            8: 9,
            9: 10,
            10: 11,
            11: 12,
        },
        "flip_caption_input": {
            2: False,
            5: True,
            7: True,
            6: False,
            4: False,
            8: False,
        },
    },
    "L": {
        "camera_id_extractor": {
            "delimiter": "-",
            "index": 1,
        },
        "camera_id_mapping_cosmos": {
            2: 0,
            4: 1,
            5: 2,
            6: 3,
            7: 4,
            8: 5,
        },
        "camera_name_mapping_cosmos": {
            2: "camera_front_wide_120fov",
            4: "camera_cross_left_120fov",
            5: "camera_cross_right_120fov",
            6: "camera_rear_tele_30fov",
            7: "camera_rear_left_70fov",
            8: "camera_rear_right_70fov",
        },
        "all_timestamp_files": [
            "timestamp_l1.csv",
        ],
        "camera_id_for_vri_caption": [2],
        "align_every_frame": False,
    },
}

SQLITE_DB_NAME = "log_0.db"
