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
"""Preview generation stage."""

import pathlib
import subprocess

import nvtx  # type: ignore[import-untyped]
from loguru import logger

from cosmos_curator.core.interfaces.stage_interface import CuratorStage, CuratorStageResource
from cosmos_curator.core.utils.config.operation_context import make_pipeline_temporary_dir
from cosmos_curator.core.utils.infra.performance_utils import StageTimer
from cosmos_curator.pipelines.video.utils.data_model import (
    SplitPipeTask,
    Window,
)


class PreviewStage(CuratorStage):
    """Stage that generates webp previews from video clips.

    This class processes video clips through a series of steps including reading,
    generating webp previews, and writing to storage.
    """

    def __init__(
        self,
        target_fps: int = 1,
        target_height: int = 240,
        *,
        verbose: bool = False,
        log_stats: bool = False,
    ) -> None:
        """Initialize the preview generation stage.

        Args:
            target_fps: Target frames per second for preview.
            target_height: Target height for preview frames.
            verbose: Whether to print verbose logs.
            log_stats: Whether to log performance statistics.

        """
        self._target_fps = target_fps
        self._target_height = target_height
        self._timer = StageTimer(self)
        self._verbose = verbose
        self._log_stats = log_stats

    @property
    def resources(self) -> CuratorStageResource:
        """Get the resource requirements for this stage.

        Returns:
            Resource configuration for the stage.

        """
        return CuratorStageResource(cpus=4.0)

    def _generate_preview(self, window: Window) -> None:
        """Generate webp preview for a video window.

        Args:
            window: Window containing video data to generate preview for.

        """
        with make_pipeline_temporary_dir(sub_dir="preview") as tmp_dir:
            input_mp4 = pathlib.Path(tmp_dir, "input.mp4")

            mp4_data = window.mp4_bytes.resolve()
            if mp4_data is None:
                msg = f"Window [{window.start_frame}, {window.end_frame}] has no mp4 bytes for preview generation."
                raise RuntimeError(msg)
            input_mp4.write_bytes(mp4_data)
            # TODO: add `del mp4_data` here to release resolved bytes (~1-10 MB)
            #   before ffmpeg runs. Currently held for the full subprocess duration.
            output_webp = pathlib.Path(tmp_dir, "output.webp")
            command = [
                "ffmpeg",
                "-threads",
                str(int(self.resources.cpus)),
                "-y",
                "-i",
                input_mp4.as_posix(),
                "-loglevel",
                "error",
                "-vf",
                f"fps={self._target_fps},scale=-1:{self._target_height}",
                "-c:v",
                "libwebp",
                "-lossless",
                str(0),
                "-compression_level",
                str(6),
                "-q:v",
                str(50),
                "-loop",
                "0",
                "-threads",
                str(int(self.resources.cpus)),
                output_webp.as_posix(),
            ]

            try:
                output = subprocess.check_output(command, stderr=subprocess.STDOUT)  # noqa: S603
                if output:
                    logger.warning(f"ffmpeg output: {output.decode('utf-8')}")
            except subprocess.CalledProcessError as e:
                # TODO: raise instead of swallowing - let process_data handle
                #   fault isolation. Currently this silently returns with no
                #   webp_bytes and no error recorded on clip.errors.
                logger.error(f"ffmpeg command failed with return code {e.returncode}")
                logger.warning(f"ffmpeg command: {' '.join(command)}")
                if e.output:
                    logger.warning(f"ffmpeg output: {e.output.decode('utf-8')}")
                return

            window.webp_bytes = output_webp.read_bytes()  # type: ignore[assignment]

    @nvtx.annotate("PreviewStage")  # type: ignore[untyped-decorator]
    def process_data(self, tasks: list[SplitPipeTask]) -> list[SplitPipeTask] | None:
        """Process video data to generate previews.

        Args:
            tasks: Tasks containing videos to process.

        Returns:
            Processed tasks with generated previews.

        """
        # TODO: add per-window fault isolation (try/except around _generate_preview)
        #   and record errors on clip.errors[f"preview_{wi}"].
        #   Currently a single window failure (e.g. mp4_bytes.resolve() raises,
        #   or mp4_data is None) crashes the entire process_data call,
        #   losing all windows/clips in the batch.
        for task in tasks:
            self._timer.reinit(self, task.get_major_size())
            video = task.video
            for clip in video.clips:
                with self._timer.time_process():
                    for window in clip.windows:
                        self._generate_preview(window)

            if self._log_stats:
                stage_name, stage_perf_stats = self._timer.log_stats()
                task.stage_perf[stage_name] = stage_perf_stats

        return tasks
