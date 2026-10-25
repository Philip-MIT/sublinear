import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc
import torch
from types import SimpleNamespace
import copy
import imageio_ffmpeg


import json
import math
import mimetypes
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, Sequence

import numpy as np
import requests
import cv2
import sys
from functools import lru_cache

SUPABASE_FUNCTIONS_URL = (
    "https://qttoxuyjvehacmihvkzu.supabase.co/functions/v1"
)

VideoData = bytes | bytearray | memoryview | BinaryIO



def _to_numpy(value: Any) -> np.ndarray:
    """Convert NumPy-compatible values and PyTorch tensors to NumPy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    return np.asarray(value)


def _normalize_video_array(
    value: Any,
    *,
    input_name: str,
    verbose: bool = False,
) -> list[np.ndarray]:
    """
    Normalize a 4D or 5D NumPy array/tensor into a list of THWC videos.

    Accepted layouts:
        T, C, H, W
        T, H, W, C
        B, T, C, H, W
        B, T, H, W, C

    C must equal 3.
    """
    array = _to_numpy(value)

    if array.ndim not in (4, 5):
        raise ValueError(
            f"{input_name} must be 4D or 5D; "
            f"received shape {array.shape}."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(
            f"{input_name} cannot contain an empty batch or video."
        )

    if array.ndim == 4:
        channel_first = array.shape[1] == 3
        channel_last = array.shape[-1] == 3

        if channel_first == channel_last:
            raise ValueError(
                f"{input_name} has an ambiguous or missing RGB "
                f"channel in shape {array.shape}. Expected "
                "(T, C, H, W) or (T, H, W, C), with C=3."
            )

        if channel_first:
            normalized = np.moveaxis(array, 1, -1)
            layout = "(T, C, H, W)"
        else:
            normalized = array
            layout = "(T, H, W, C)"

        if verbose:
            print(
                f"{input_name}: received shape {array.shape}; "
                f"assuming one video with T={array.shape[0]} "
                f"in {layout} layout.",
                flush=True,
            )

        return [np.ascontiguousarray(normalized)]

    channel_first = array.shape[2] == 3
    channel_last = array.shape[-1] == 3

    if channel_first == channel_last:
        raise ValueError(
            f"{input_name} has an ambiguous or missing RGB "
            f"channel in shape {array.shape}. Expected "
            "(B, T, C, H, W) or (B, T, H, W, C), with C=3."
        )

    if channel_first:
        normalized = np.moveaxis(array, 2, -1)
        layout = "(B, T, C, H, W)"
    else:
        normalized = array
        layout = "(B, T, H, W, C)"

    if verbose:
        print(
            f"{input_name}: received shape {array.shape}; "
            f"assuming B={array.shape[0]} videos with "
            f"T={array.shape[1]} in {layout} layout.",
            flush=True,
        )

    return [
        np.ascontiguousarray(normalized[batch_index])
        for batch_index in range(normalized.shape[0])
    ]


def _split_frame_episodes(
    value: Any,
    *,
    input_name: str = "video_frames",
    verbose: bool = False,
) -> list[np.ndarray]:
    """
    Return a list of videos, with every video normalized to T×H×W×3.

    NumPy arrays and PyTorch tensors use explicit dimensional semantics:
      - 4D: one video
      - 5D: a batch of videos

    Lists of frames and lists of videos remain supported for backwards
    compatibility.
    """
    try:
        array = _to_numpy(value)
    except (TypeError, ValueError):
        array = None

    if (
        array is not None
        and array.dtype != object
        and array.ndim in (4, 5)
    ):
        return _normalize_video_array(
            array,
            input_name=input_name,
            verbose=verbose,
        )

    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        shape = getattr(array, "shape", None)
        raise ValueError(
            f"{input_name} must be a 4D/5D NumPy array or tensor, "
            f"or a sequence of frames/videos; received shape {shape}."
        )

    if len(value) == 0:
        raise ValueError(f"{input_name} cannot be empty.")

    first = _to_numpy(value[0])

    # A sequence of individual HWC/CHW frames is one video.
    if first.ndim == 3:
        try:
            stacked = np.stack(
                [_to_numpy(frame) for frame in value],
                axis=0,
            )
        except ValueError as error:
            raise ValueError(
                f"All frames in {input_name} must have the same shape."
            ) from error

        return _normalize_video_array(
            stacked,
            input_name=input_name,
            verbose=verbose,
        )

    # Otherwise, treat each item as one video.
    episodes: list[np.ndarray] = []

    for episode_index, episode in enumerate(value):
        normalized = _normalize_video_array(
            episode,
            input_name=f"{input_name}[{episode_index}]",
            verbose=verbose,
        )

        if len(normalized) != 1:
            raise ValueError(
                f"{input_name}[{episode_index}] must be one 4D "
                "video, not another batch."
            )

        episodes.append(normalized[0])

    return episodes


def _classify_view_type(value: str) -> str | None:
    """
    Classify a caller-provided view label.

    Static synonyms: top, static, external, side
    Wrist labels: wrist, left_wrist, right_wrist
    """
    normalized = "".join(
        character if character.isalnum() else "_"
        for character in str(value).strip().lower()
    )
    tokens = {
        token
        for token in normalized.split("_")
        if token
    }

    is_static = bool(
        tokens.intersection(
            {"top", "static", "external", "side"}
        )
    )
    is_wrist = "wrist" in tokens
    is_left = "left" in tokens
    is_right = "right" in tokens

    # Labels such as "external_and_wrist" are ambiguous.
    if is_static and is_wrist:
        return None

    if is_wrist:
        if is_left and is_right:
            return None
        if is_left:
            return "left_wrist"
        if is_right:
            return "right_wrist"
        return "wrist"

    if is_static:
        return "static"

    return None


def _view_stitch_order(
    view_types: str | Sequence[str] | None,
    view_count: int,
) -> list[int]:
    """
    Return indices in their desired left-to-right order.

    Three recognized views:
        left_wrist, static, right_wrist

    Two recognized views:
        static, wrist

    Unknown, ambiguous, incomplete, or duplicated classifications
    preserve the original input order.
    """
    original_order = list(range(view_count))

    if view_types is None:
        return original_order

    if isinstance(view_types, str):
        values = [view_types]
    else:
        values = list(view_types)

    if len(values) != view_count:
        return original_order

    categories = [
        _classify_view_type(value)
        for value in values
    ]

    if view_count == 3:
        desired = [
            "left_wrist",
            "static",
            "right_wrist",
        ]

        if all(categories.count(category) == 1 for category in desired):
            return [
                categories.index(category)
                for category in desired
            ]

    elif view_count == 2:
        simplified = [
            (
                "wrist"
                if category
                in {"left_wrist", "right_wrist", "wrist"}
                else category
            )
            for category in categories
        ]
        desired = ["static", "wrist"]

        if all(simplified.count(category) == 1 for category in desired):
            return [
                simplified.index(category)
                for category in desired
            ]

    return original_order


def _roles(
    view_type: str | Sequence[str] | None,
    count: int,
) -> list[str]:
    if view_type is None:
        # Keep the existing single-view default. For multiple unknown
        # views, use neutral names so no ordering is invented.
        if count == 1:
            return ["static"]

        return [
            f"view{index + 1}"
            for index in range(count)
        ]

    roles = (
        [view_type]
        if isinstance(view_type, str)
        else list(view_type)
    )

    if len(roles) != count or any(
        not str(role).strip()
        for role in roles
    ):
        raise ValueError(
            "view_type must contain exactly one non-empty "
            "name per view."
        )

    return [
        str(role).strip()
        for role in roles
    ]


def _ordered_lerobot_view_map(
    view_map: Mapping[str, str],
) -> dict[str, str]:
    """
    Order the mapping by its role keys. Python dict insertion order
    is retained when the roles cannot be classified confidently.
    """
    items = list(view_map.items())
    order = _view_stitch_order(
        [role for role, _camera_key in items],
        len(items),
    )

    return {
        items[index][0]: items[index][1]
        for index in order
    }


def _frame_to_rgb_uint8(frame: Any) -> np.ndarray:
    frame = _to_numpy(frame)

    if (
        frame.ndim == 3
        and frame.shape[0] == 3
        and frame.shape[-1] != 3
    ):
        frame = np.moveaxis(frame, 0, -1)

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            "Every frame must have shape H×W×3 or C×H×W "
            f"with C=3; received {frame.shape}."
        )

    if np.issubdtype(frame.dtype, np.floating):
        frame = np.nan_to_num(frame)

        if (
            frame.size
            and frame.min() >= 0
            and frame.max() <= 1
        ):
            frame = frame * 255

    return np.ascontiguousarray(
        np.clip(frame, 0, 255).astype(np.uint8)
    )


def _stitch_frame_lists(
    frame_lists: Sequence[Sequence[Any]],
) -> list[np.ndarray]:
    """
    Stitch already ordered views horizontally into the final frame_list.
    """
    if not frame_lists:
        raise ValueError("At least one frame list is required.")

    frame_counts = [
        len(frames)
        for frames in frame_lists
    ]

    if len(set(frame_counts)) != 1:
        raise ValueError(
            "All views must have the same frame count; "
            f"received {frame_counts}."
        )

    result: list[np.ndarray] = []

    for frames_at_timestep in zip(*frame_lists):
        frames = [
            _frame_to_rgb_uint8(frame)
            for frame in frames_at_timestep
        ]

        target_height = min(
            frame.shape[0]
            for frame in frames
        )
        resized_frames: list[np.ndarray] = []

        for frame in frames:
            if frame.shape[0] == target_height:
                resized_frames.append(frame)
                continue

            target_width = round(
                frame.shape[1]
                * target_height
                / frame.shape[0]
            )
            resized_frames.append(
                cv2.resize(
                    frame,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            )

        result.append(
            np.ascontiguousarray(
                np.concatenate(resized_frames, axis=1)
            )
        )

    return result


def _as_path_list(value: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(value, (str, Path)):
        paths = [Path(value)]
    else:
        paths = [Path(path) for path in value]

    if not paths:
        raise ValueError("Video path input cannot be empty.")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    return paths

def _video_path_info(path: Path) -> tuple[int, float | None]:
    """
    Return the exact number of decodable frames and the reported FPS.

    Counting with grab() avoids retaining decoded frames in memory and is
    more reliable than trusting CAP_PROP_FRAME_COUNT for every container.
    """
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = 0

    try:
        while capture.grab():
            frame_count += 1
    finally:
        capture.release()

    if frame_count < 1:
        raise ValueError(f"Video contains no decodable frames: {path}")

    source_fps = (
        reported_fps
        if math.isfinite(reported_fps) and reported_fps > 0
        else None
    )

    return frame_count, source_fps


def _read_video_frames_at_indices(
    path: Path,
    source_frame_indices: Sequence[int],
) -> list[np.ndarray]:
    """Decode only the requested, strictly increasing frame indices."""
    indices = [int(index) for index in source_frame_indices]

    if not indices:
        raise ValueError("source_frame_indices cannot be empty.")

    if indices[0] < 0 or any(
        left >= right
        for left, right in zip(indices, indices[1:])
    ):
        raise ValueError(
            "source_frame_indices must be non-negative and "
            "strictly increasing."
        )

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    frames: list[np.ndarray] = []
    target_position = 0
    frame_index = 0

    try:
        while target_position < len(indices):
            ok, frame = capture.read()

            if not ok:
                break

            if frame_index == indices[target_position]:
                frames.append(frame)
                target_position += 1

            frame_index += 1
    finally:
        capture.release()

    if len(frames) != len(indices):
        raise ValueError(
            f"Unable to read every requested frame from {path}; "
            f"read {len(frames)} of {len(indices)}."
        )

    return frames


_FFMPEG_TIMEOUT_SECONDS = 1800
_DEFAULT_MAX_VIDEOS_PER_JOB = 30
_BACKEND_MAX_VIDEOS_PER_JOB = 100
_MISSING = object()


def _resolve_binary(value: str | Path | None, default: str) -> str:
    requested = str(value) if value is not None else default
    resolved = shutil.which(requested)

    if resolved is None and Path(requested).is_file():
        resolved = str(Path(requested).resolve())

    if resolved is None:
        raise RuntimeError(
            f"{default} is required for the LeRobot video fast path; "
            f"could not resolve {requested!r}."
        )

    return resolved


def _run_tool(args: list[str], *, description: str) -> str:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{description} timed out") from error
    except OSError as error:
        raise RuntimeError(
            f"Unable to run {args[0]!r} for {description}"
        ) from error

    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise RuntimeError(
            f"{description} failed with exit code "
            f"{result.returncode}:\n{detail}"
        )

    return result.stdout

@lru_cache(maxsize=None)
def _ffmpeg_passthrough_args(
    ffmpeg: str,
) -> tuple[str, str]:
    """
    Return the frame-synchronization arguments supported by this
    specific FFmpeg binary.
    """
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-h",
                "full",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Timed out while inspecting FFmpeg capabilities."
        ) from error
    except OSError as error:
        raise RuntimeError(
            f"Unable to inspect FFmpeg binary {ffmpeg!r}."
        ) from error

    help_text = (
        (result.stdout or "")
        + "\n"
        + (result.stderr or "")
    )

    # Preferred by current FFmpeg versions.
    if "-fps_mode" in help_text:
        return ("-fps_mode", "passthrough")

    # Compatible with older FFmpeg versions.
    if "-vsync" in help_text:
        return ("-vsync", "0")

    raise RuntimeError(
        "This FFmpeg build supports neither -fps_mode nor "
        "-vsync; frame-preserving video sampling is unavailable."
    )

def _episode_metadata(meta: Any, episode_index: int) -> dict[str, Any]:
    """Return an episode metadata row without reading frame parquet."""
    ensure_readable = getattr(meta, "ensure_readable", None)
    if callable(ensure_readable):
        ensure_readable()

    table = getattr(meta, "episodes", None)
    if table is None:
        raise TypeError("LeRobot episode metadata is not loaded.")

    row: Any = _MISSING

    if isinstance(table, Mapping):
        row = table.get(
            episode_index,
            table.get(str(episode_index), _MISSING),
        )

        # Defensive support for a dict-of-columns representation.
        if row is _MISSING and "episode_index" in table:
            try:
                row = {
                    key: column[episode_index]
                    for key, column in table.items()
                }
            except (IndexError, KeyError, TypeError):
                row = _MISSING
    else:
        # Official v3 metadata keeps global episode rows ordered by index.
        try:
            row = table[episode_index]
        except (IndexError, KeyError, TypeError):
            row = _MISSING

    if row is _MISSING or not isinstance(row, Mapping):
        raise ValueError(
            f"No metadata exists for LeRobot episode {episode_index}."
        )

    result = dict(row)
    recorded_index = result.get("episode_index")

    if (
        recorded_index is not None
        and int(recorded_index) != episode_index
    ):
        raise ValueError(
            f"Episode metadata row {episode_index} describes episode "
            f"{int(recorded_index)} instead."
        )

    return result


def _selected_lerobot_episode_indices(
    dataset: Any,
    episodes: int | Sequence[int] | None,
    *,
    limit: int = 100,
) -> list[int]:
    """Validate selection using episode metadata, never frame parquet."""
    meta = getattr(dataset, "meta", None)

    if meta is None:
        raise TypeError(
            "lerobot_dataset must be a loaded local LeRobotDataset."
        )

    try:
        total = int(meta.total_episodes)
    except Exception as error:
        raise TypeError(
            "LeRobot total_episodes metadata is unavailable."
        ) from error

    loaded = getattr(dataset, "episodes", None)
    allowed: set[int] | None = None
    default_selection: list[int] | None = None

    if loaded is not None:
        try:
            default_selection = [int(value) for value in loaded]
        except TypeError as error:
            raise TypeError(
                "Unexpected LeRobot dataset.episodes value."
            ) from error

        allowed = set(default_selection)

    if episodes is None:
        if default_selection is not None:
            selected = default_selection
        else:
            if total > limit:
                raise ValueError(
                    f"Dataset contains {total} episodes; explicitly "
                    f"select at most {limit}."
                )

            selected = list(range(total))
    elif (
        isinstance(episodes, Integral)
        and not isinstance(episodes, bool)
    ):
        selected = [int(episodes)]
    else:
        if isinstance(episodes, (str, bytes, bytearray)):
            raise TypeError(
                "episodes must be an integer or a sequence of integers."
            )

        selected = []

        for value in episodes:
            if (
                not isinstance(value, Integral)
                or isinstance(value, bool)
            ):
                raise TypeError(
                    "episodes must contain integer episode indices."
                )

            selected.append(int(value))

    if not selected:
        raise ValueError("At least one episode must be selected.")

    if len(selected) > limit:
        raise ValueError(
            f"A request may contain at most {limit} episodes."
        )

    if len(set(selected)) != len(selected):
        raise ValueError(
            "episodes must not contain duplicate indices."
        )

    missing = [
        index
        for index in selected
        if (
            index < 0
            or index >= total
            or (allowed is not None and index not in allowed)
        )
    ]

    if missing:
        raise ValueError(
            "Unknown or unavailable LeRobot episode indices: "
            f"{missing}"
        )

    # Also catches sparse/corrupt metadata without reading video frames.
    for index in selected:
        _episode_metadata(meta, index)

    return selected


def _lerobot_video_keys(meta: Any) -> set[str]:
    """Return only MP4-backed keys; camera_keys also includes images."""
    video_keys = {
        str(key)
        for key in (getattr(meta, "video_keys", ()) or ())
    }
    features = getattr(meta, "features", None)

    if isinstance(features, Mapping):
        for key, feature in features.items():
            if isinstance(feature, Mapping):
                dtype = feature.get("dtype")
            else:
                dtype = getattr(feature, "dtype", None)

            normalized = str(dtype).lower()
            if normalized == "video" or normalized.endswith(".video"):
                video_keys.add(str(key))

    return video_keys


def _metadata_codebase_major(meta: Any) -> int | None:
    info = getattr(meta, "info", None)
    raw_version = getattr(info, "codebase_version", None)

    if raw_version is None and isinstance(info, Mapping):
        raw_version = info.get("codebase_version")
    elif raw_version is None and hasattr(info, "get"):
        raw_version = info.get("codebase_version")

    try:
        return int(str(raw_version).lstrip("v").split(".", 1)[0])
    except (TypeError, ValueError):
        return None


def _metadata_is_v2_episode_file_layout(meta: Any) -> bool:
    template = str(getattr(meta, "video_path", "") or "")

    if "{episode_index" in template:
        return True

    major = _metadata_codebase_major(meta)
    return major is not None and major < 3


def _episode_clip_spec(
    row: Mapping[str, Any],
    camera_key: str,
    source_fps: float,
) -> tuple[float, float, int, bool]:
    length_value = row.get("length")

    if length_value is None:
        start_index = row.get("dataset_from_index")
        stop_index = row.get("dataset_to_index")

        if start_index is not None and stop_index is not None:
            length_value = int(stop_index) - int(start_index)

    start_value = row.get(
        f"videos/{camera_key}/from_timestamp"
    )
    stop_value = row.get(
        f"videos/{camera_key}/to_timestamp"
    )

    if (start_value is None) != (stop_value is None):
        raise ValueError(
            f"Incomplete timestamp metadata for {camera_key}."
        )

    explicit_bounds = start_value is not None

    if explicit_bounds:
        start = float(start_value)
        stop = float(stop_value)
        bounds_frames = (
            round(stop * source_fps)
            - round(start * source_fps)
        )

        if length_value is None:
            length_value = bounds_frames
        elif int(length_value) != bounds_frames:
            raise ValueError(
                f"Timestamp range for {camera_key} contains "
                f"{bounds_frames} frames at {source_fps:g} fps, "
                f"but episode metadata says {int(length_value)}."
            )
    else:
        start = 0.0

        if length_value is None:
            raise ValueError(
                "Episode metadata has no frame length."
            )

        stop = int(length_value) / source_fps

    expected_frames = int(length_value)

    if expected_frames < 1:
        raise ValueError(
            "Episode metadata has no positive frame length."
        )

    if (
        not math.isfinite(start)
        or not math.isfinite(stop)
        or start < 0
        or stop <= start
    ):
        raise ValueError(
            f"Invalid video interval [{start}, {stop}) for "
            f"{camera_key}."
        )

    return start, stop, expected_frames, explicit_bounds


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _probe_video(
    ffprobe: str,
    path: Path,
    *,
    count_packets: bool,
) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
    ]

    if count_packets:
        command.append("-count_packets")

    command += [
        "-show_entries",
        (
            "stream=codec_name,width,height,start_time,duration,"
            "nb_frames,nb_read_packets:format=duration"
        ),
        "-of",
        "json",
        str(path),
    ]

    raw = _run_tool(command, description=f"probing {path}")

    try:
        data = json.loads(raw)
        stream = data["streams"][0]
    except (
        json.JSONDecodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as error:
        raise RuntimeError(
            f"ffprobe found no readable video stream in {path}"
        ) from error

    stream_duration = _optional_float(stream.get("duration"))
    format_duration = _optional_float(
        (data.get("format") or {}).get("duration")
    )
    packet_count = (
        _optional_int(stream.get("nb_read_packets"))
        if count_packets
        else None
    )

    return {
        "codec": stream.get("codec_name"),
        "width": _optional_int(stream.get("width")),
        "height": _optional_int(stream.get("height")),
        "start_time": _optional_float(stream.get("start_time")),
        "duration": (
            stream_duration
            if stream_duration is not None
            else format_duration
        ),
        "packet_count": packet_count,
        "declared_frame_count": _optional_int(
            stream.get("nb_frames")
        ),
    }


def _validate_episode_video(
    ffprobe: str,
    path: Path,
    *,
    expected_frames: int,
    expected_duration: float,
    source_fps: float,
) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Video output is missing or empty: {path}"
        )

    info = _probe_video(ffprobe, path, count_packets=True)
    errors: list[str] = []

    if not info["width"] or not info["height"]:
        errors.append("invalid dimensions")

    counted_frames = info["packet_count"]

    if counted_frames is None:
        counted_frames = info["declared_frame_count"]

    if counted_frames is None:
        errors.append("ffprobe did not report a packet/frame count")
    elif counted_frames != expected_frames:
        errors.append(
            f"contains {counted_frames} video packets/frames; "
            f"expected {expected_frames}"
        )

    duration_tolerance = max(0.1, 2.0 / source_fps)

    if info["duration"] is None:
        errors.append("ffprobe did not report a duration")
    elif (
        abs(info["duration"] - expected_duration)
        > duration_tolerance
    ):
        errors.append(
            f"duration is {info['duration']:.6f}s; expected "
            f"{expected_duration:.6f}s +/- "
            f"{duration_tolerance:.6f}s"
        )

    if errors:
        raise RuntimeError(
            f"Invalid episode video {path}: "
            + "; ".join(errors)
        )


def _has_keyframe_at(
    ffprobe: str,
    source: Path,
    *,
    timestamp: float,
    source_fps: float,
) -> bool:
    """Check that stream-copy clipping can start at this boundary."""
    tolerance = max(0.001, 0.51 / source_fps)

    if timestamp <= tolerance:
        # A valid standalone MP4 starts with a random-access frame.
        return True

    window = max(1.0, 4.0 / source_fps)
    interval = (
        f"{max(0.0, timestamp - window):.9f}%"
        f"{timestamp + window:.9f}"
    )
    raw = _run_tool(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            interval,
            "-show_packets",
            "-show_entries",
            "packet=pts_time,dts_time,flags",
            "-of",
            "json",
            str(source),
        ],
        description=f"checking the keyframe boundary in {source}",
    )

    try:
        packets = json.loads(raw).get("packets", [])
    except (json.JSONDecodeError, AttributeError) as error:
        raise RuntimeError(
            f"Unable to inspect keyframes in {source}"
        ) from error

    for packet in packets:
        if "K" not in str(packet.get("flags", "")):
            continue

        packet_time = _optional_float(packet.get("pts_time"))

        if packet_time is None:
            packet_time = _optional_float(packet.get("dts_time"))

        if (
            packet_time is not None
            and abs(packet_time - timestamp) <= tolerance
        ):
            return True

    return False


def _remux_episode(
    ffmpeg: str,
    source: Path,
    output: Path,
    *,
    start: float,
    stop: float,
    expected_frames: int,
) -> None:
    output.unlink(missing_ok=True)
    _run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.9f}",
            "-i",
            str(source),
            "-t",
            f"{stop - start:.9f}",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "copy",
            "-copytb",
            "1",
            "-frames:v",
            str(expected_frames),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(output),
        ],
        description=(
            f"stream-copying [{start:.6f}, {stop:.6f}) "
            f"from {source}"
        ),
    )



def _transcode_episode(
    ffmpeg: str,
    source: Path,
    output: Path,
    *,
    start: float,
    stop: float,
    expected_frames: int,
) -> None:
    output.unlink(missing_ok=True)
    _run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            # Input seeking is frame-accurate while transcoding: FFmpeg
            # decodes and discards any preceding GOP frames.
            "-ss",
            f"{start:.9f}",
            "-i",
            str(source),
            "-t",
            f"{stop - start:.9f}",
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-frames:v",
            str(expected_frames),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(output),
        ],
        description=(
            f"transcoding [{start:.6f}, {stop:.6f}) "
            f"from {source}"
        ),
    )

def _uniform_frame_indices(
    source_frame_count: int,
    max_frames: int,
) -> list[int]:
    if source_frame_count < 1:
        raise ValueError("source_frame_count must be positive.")
    if max_frames < 1:
        raise ValueError("max_frames must be positive.")

    output_frame_count = min(source_frame_count, max_frames)

    if output_frame_count == 1:
        return [0]

    # Exact integer equivalent of uniformly sampling both endpoints.
    return [
        position * (source_frame_count - 1)
        // (output_frame_count - 1)
        for position in range(output_frame_count)
    ]


def _sampled_fps(
    source_fps: float,
    source_frame_count: int,
    output_frame_count: int,
) -> float:
    if source_frame_count > 1 and output_frame_count > 1:
        return (
            source_fps
            * (output_frame_count - 1)
            / (source_frame_count - 1)
        )

    return source_fps


def _transcode_sampled_episode(
    ffmpeg: str,
    source: Path,
    output: Path,
    *,
    start: float,
    source_frame_indices: Sequence[int],
    output_fps: float,
) -> None:
    output.unlink(missing_ok=True)

    select_expression = "+".join(
        f"eq(n\\,{frame_index})"
        for frame_index in source_frame_indices
    )

    _run_tool(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{start:.9f}",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            (
                f"select={select_expression},"
                f"setpts=N/({output_fps:.17g}*TB)"
            ),
            *_ffmpeg_passthrough_args(ffmpeg),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-frames:v",
            str(len(source_frame_indices)),
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-avoid_negative_ts",
            "make_zero",
            "-movflags",
            "+faststart",
            str(output),
        ],
        description=(
            f"sampling {len(source_frame_indices)} frames "
            f"from {source}"
        ),
    )

def _write_lerobot_videos_into_archive(
    *,
    archive: zipfile.ZipFile,
    bundle_path: Path,
    dataset: Any,
    episodes: int | Sequence[int] | None,
    view_map: dict[str, str] | None,
    requested_fps: float | None,
    ffmpeg_path: str | Path | None,
    ffprobe_path: str | Path | None,
    remux_fallback: str,
    max_frames: int | None,
    max_videos: int,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], float]:
    """Package local MP4-backed LeRobot episodes without decoding."""
    selected = _selected_lerobot_episode_indices(
        dataset,
        episodes,
        limit=max_videos,
    )
    meta = dataset.meta
    video_keys = _lerobot_video_keys(meta)
    depth_keys = set(getattr(meta, "depth_keys", ()) or ())

    if view_map is None:
        defaults = {
            "static": "observation.images.top",
            "left_wrist": "observation.images.left_wrist",
            "right_wrist": "observation.images.right_wrist",
        }
        view_map = {
            role: camera_key
            for role, camera_key in defaults.items()
            if camera_key in video_keys
        }
    else:
        view_map = dict(view_map)

    if not view_map:
        raise ValueError(
            "No default video-backed LeRobot camera keys were found. "
            "Pass lerobot_view_map="
            "{'static': 'camera.key', ...}."
        )

    if any(not str(role).strip() for role in view_map):
        raise ValueError(
            "LeRobot view roles must be non-empty strings."
        )
    
    view_map = _ordered_lerobot_view_map(view_map)

    non_video = [
        key for key in view_map.values() if key not in video_keys
    ]

    if non_video:
        raise ValueError(
            "The fast LeRobot path requires video-backed features; "
            "these keys are image-backed or unknown: "
            f"{non_video}. Convert them to LeRobot video storage or "
            "pass explicit MP4 paths."
        )

    requested_depth = [
        key for key in view_map.values() if key in depth_keys
    ]

    if requested_depth:
        raise ValueError(
            "Quantized LeRobot depth videos are not RGB backend "
            f"inputs: {requested_depth}"
        )

    if getattr(dataset, "image_transforms", None) is not None:
        raise ValueError(
            "The compressed LeRobot fast path cannot apply "
            "image_transforms. Pass a dataset without transforms or "
            "supply already-transformed video frames."
        )

    delta_timestamps = getattr(dataset, "delta_timestamps", None)

    if isinstance(delta_timestamps, Mapping):
        delta_video_keys = [
            key
            for key in view_map.values()
            if key in delta_timestamps
            and delta_timestamps[key] is not None
        ]
    else:
        delta_video_keys = (
            list(view_map.values()) if delta_timestamps else []
        )

    if delta_video_keys:
        raise ValueError(
            "The compressed LeRobot fast path cannot materialize "
            f"delta-timestamp video stacks: {delta_video_keys}"
        )

    meta_fps = getattr(meta, "fps", None)

    if meta_fps is None:
        meta_fps = getattr(dataset, "fps", None)

    if meta_fps is None:
        raise ValueError("LeRobot fps metadata is unavailable.")

    source_fps = float(meta_fps)

    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid LeRobot fps: {source_fps!r}")

    if requested_fps is not None:
        requested_fps = float(requested_fps)

        if (
            not math.isfinite(requested_fps)
            or requested_fps <= 0
        ):
            raise ValueError(f"Invalid requested fps: {requested_fps!r}")

        if not math.isclose(
            requested_fps,
            source_fps,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "fps cannot be changed by the LeRobot remux fast "
                f"path; dataset fps is {source_fps:g}, requested "
                f"{requested_fps:g}."
            )

    if remux_fallback not in {"error", "transcode"}:
        raise ValueError(
            "lerobot_remux_fallback must be 'error' or "
            "'transcode'."
        )

    resolved_ffmpeg: str | None = None
    resolved_ffprobe: str | None = None

    def require_ffmpeg() -> str:
        nonlocal resolved_ffmpeg

        if resolved_ffmpeg is None:
            resolved_ffmpeg = _resolve_binary(
                ffmpeg_path,
                "ffmpeg",
            )

        return resolved_ffmpeg

    def require_ffprobe() -> str:
        nonlocal resolved_ffprobe

        if resolved_ffprobe is None:
            resolved_ffprobe = _resolve_binary(
                ffprobe_path,
                "ffprobe",
            )

        return resolved_ffprobe

    dataset_root = Path(dataset.root)
    keyframe_cache: dict[tuple[Path, float], bool] = {}
    direct_source_owners: dict[tuple[str, Path], int] = {}
    episode_manifests: list[dict[str, Any]] = []
    prepared: list[dict[str, Any]] = []

    # Prepare and validate every clip before writing any ZIP member.
    with tempfile.TemporaryDirectory(
        prefix="sublinear-lerobot-",
        dir=bundle_path.parent,
    ) as clips_directory:
        clips_root = Path(clips_directory)

        for episode_index in selected:
            row = _episode_metadata(meta, episode_index)
            episode_manifest: dict[str, Any] = {
                "episode_index": episode_index,
                "views": [],
            }

            # These values must reset for every episode.
            episode_source_frame_count: int | None = None
            sampled_indices: list[int] | None = None
            output_fps = source_fps

            for view_position, (role, camera_key) in enumerate(
                view_map.items()
            ):
                (
                    start,
                    stop,
                    expected_frames,
                    explicit_bounds,
                ) = _episode_clip_spec(
                    row,
                    camera_key,
                    source_fps,
                )
                if episode_source_frame_count is None:
                    episode_source_frame_count = expected_frames

                    sampling: dict[str, Any] = {
                        "source_frame_count": expected_frames,
                    }

                    dataset_from_index = _optional_int(
                        row.get("dataset_from_index")
                    )
                    if dataset_from_index is not None:
                        sampling["dataset_from_index"] = dataset_from_index

                    if (
                        max_frames is not None
                        and expected_frames > max_frames
                    ):
                        sampled_indices = _uniform_frame_indices(
                            expected_frames,
                            max_frames,
                        )
                        sampling["source_frame_indices"] = sampled_indices
                        output_fps = _sampled_fps(
                            source_fps,
                            expected_frames,
                            len(sampled_indices),
                        )

                    episode_manifest["fps"] = output_fps
                    episode_manifest["sampling"] = sampling

                elif expected_frames != episode_source_frame_count:
                    raise ValueError(
                        f"Episode {episode_index} views have different frame "
                        f"counts: {episode_source_frame_count} and "
                        f"{expected_frames}."
                    )
                relative_source = Path(
                    meta.get_video_file_path(
                        episode_index,
                        camera_key,
                    )
                )
                source = (
                    relative_source
                    if relative_source.is_absolute()
                    else dataset_root / relative_source
                ).resolve()

                if (
                    not source.is_file()
                    or source.stat().st_size == 0
                ):
                    raise FileNotFoundError(
                        "Missing LeRobot video for episode "
                        f"{episode_index}, key {camera_key}: "
                        f"{source}"
                    )

                output = clips_root / (
                    f"episode_{episode_index}_"
                    f"view_{view_position}.mp4"
                )
                payload_path: Path | None = None
                packaging_mode: str | None = None
                fast_path_error: Exception | None = None

                if sampled_indices is not None:
                    _transcode_sampled_episode(
                        require_ffmpeg(),
                        source,
                        output,
                        start=start,
                        source_frame_indices=sampled_indices,
                        output_fps=output_fps,
                    )
                    _validate_episode_video(
                        require_ffprobe(),
                        output,
                        expected_frames=len(sampled_indices),
                        expected_duration=len(sampled_indices) / output_fps,
                        source_fps=output_fps,
                    )
                    payload_path = output
                    packaging_mode = "sampled_transcode"

                elif not explicit_bounds:
                    if not _metadata_is_v2_episode_file_layout(meta):
                        raise ValueError(
                            f"{camera_key} has no episode timestamps, "
                            "but the metadata does not describe a v2 "
                            "per-episode layout; refusing to upload a "
                            "whole shared shard."
                        )

                    owner_key = (camera_key, source)
                    previous_owner = direct_source_owners.get(owner_key)

                    if (
                        previous_owner is not None
                        and previous_owner != episode_index
                    ):
                        raise ValueError(
                            f"Episodes {previous_owner} and "
                            f"{episode_index} resolve to the same "
                            f"supposedly per-episode video: {source}"
                        )

                    direct_source_owners[owner_key] = episode_index
                    payload_path = source
                    packaging_mode = "copy"
                else:
                    try:
                        boundary_key = (source, round(start, 9))
                        on_keyframe = keyframe_cache.get(boundary_key)

                        if on_keyframe is None:
                            on_keyframe = _has_keyframe_at(
                                require_ffprobe(),
                                source,
                                timestamp=start,
                                source_fps=source_fps,
                            )
                            keyframe_cache[boundary_key] = on_keyframe

                        if not on_keyframe:
                            raise RuntimeError(
                                f"Episode {episode_index} start "
                                f"{start:.6f}s is not a keyframe in "
                                f"{source}; exact stream-copy clipping "
                                "is unsafe."
                            )

                        _remux_episode(
                            require_ffmpeg(),
                            source,
                            output,
                            start=start,
                            stop=stop,
                            expected_frames=expected_frames,
                        )
                        _validate_episode_video(
                            require_ffprobe(),
                            output,
                            expected_frames=expected_frames,
                            expected_duration=stop - start,
                            source_fps=source_fps,
                        )
                        payload_path = output
                        packaging_mode = "remux"
                    except Exception as error:
                        fast_path_error = error

                if payload_path is None:
                    if remux_fallback == "error":
                        raise RuntimeError(
                            "Unable to package episode "
                            f"{episode_index}, view {camera_key} "
                            "without transcoding."
                        ) from fast_path_error

                    try:
                        _transcode_episode(
                            require_ffmpeg(),
                            source,
                            output,
                            start=start,
                            stop=stop,
                            expected_frames=expected_frames,
                        )
                        _validate_episode_video(
                            require_ffprobe(),
                            output,
                            expected_frames=expected_frames,
                            expected_duration=stop - start,
                            source_fps=source_fps,
                        )
                    except Exception as transcode_error:
                        raise RuntimeError(
                            "Both the compressed-video fast path and "
                            "fallback transcode failed for episode "
                            f"{episode_index}, view {camera_key}. "
                            f"Fast-path error: {fast_path_error}"
                        ) from transcode_error

                    payload_path = output
                    packaging_mode = "transcode"

                member = (
                    f"episodes/{episode_index}/"
                    f"view_{view_position}.mp4"
                )
                prepared.append(
                    {
                        "payload_path": payload_path,
                        "member": member,
                        "episode_index": episode_index,
                        "camera_key": camera_key,
                        "packaging_mode": packaging_mode,
                    }
                )
                episode_manifest["views"].append(
                    {
                        "role": str(role).strip(),
                        "kind": "video",
                        "path": member,
                        "source_key": camera_key,
                        "source_video_frame_offset": int(
                            round(start * source_fps)
                        ),
                    }
                )

            episode_manifests.append(episode_manifest)

        for clip in prepared:
            payload_path = clip["payload_path"]
            archive.write(payload_path, clip["member"])
            if verbose:
                print(
                    f"Packaged episode {clip['episode_index']}, "
                    f"{clip['camera_key']} via "
                    f"{clip['packaging_mode']}: "
                    f"{payload_path.stat().st_size / 1024**2:.1f} MiB",
                    flush=True,
                )

    return episode_manifests, source_fps



def _write_bundle(
    bundle_path: Path,
    *,
    video_paths_views: list[Any],
    video_frames_views: list[Any],
    view_type: str | Sequence[str] | None,
    color_order: str | Sequence[str],
    lerobot_dataset: Any,
    episodes: int | Sequence[int] | None,
    lerobot_view_map: dict[str, str] | None,
    fps: float | None,
    max_frames: int | None = 100,
    max_videos: int = _DEFAULT_MAX_VIDEOS_PER_JOB,
    ffmpeg_path: str | Path | None = None,
    ffprobe_path: str | Path | None = None,
    lerobot_remux_fallback: str = "transcode",
    verbose: bool = False,
) -> int:
    """
    Build one .slb ZIP containing one or more episodes.

    max_frames is applied uniformly to video paths, in-memory frames,
    and video-backed LeRobot episodes.

    max_videos limits the number of episodes included in this job.
    """

    input_family_count = sum(
        [
            bool(video_paths_views),
            bool(video_frames_views),
            lerobot_dataset is not None,
        ]
    )

    if input_family_count != 1:
        raise ValueError(
            "Provide exactly one of video paths, video frames, "
            "or lerobot_dataset."
        )

    if max_frames is not None and (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
    ):
        raise ValueError(
            "max_frames must be a positive integer or None."
        )

    if (
        isinstance(max_videos, bool)
        or not isinstance(max_videos, int)
        or not 1 <= max_videos <= _BACKEND_MAX_VIDEOS_PER_JOB
    ):
        raise ValueError(
            "max_videos must be an integer between 1 and "
            f"{_BACKEND_MAX_VIDEOS_PER_JOB}."
        )

    def validate_episode_count(episode_count: int) -> None:
        if not 1 <= episode_count <= max_videos:
            raise ValueError(
                f"A job may contain 1–{max_videos} videos; "
                f"received {episode_count}."
            )

    manifest: dict[str, Any] = {
        "format": "sublinear-bundle",
        "version": 1,
        "max_frames": max_frames,
        "episodes": [],
    }

    def sampled_offsets(frame_count: int) -> np.ndarray:
        if frame_count < 1:
            raise ValueError("An episode cannot contain zero frames.")

        if max_frames is None or frame_count <= max_frames:
            return np.arange(frame_count, dtype=np.int64)

        # Includes both the first and final frame.
        return np.linspace(
            0,
            frame_count - 1,
            num=max_frames,
            dtype=np.int64,
        )

    def sampled_fps(
        source_fps: float,
        source_frame_count: int,
        output_frame_count: int,
    ) -> float:
        """
        Preserve approximately the same playback duration after sampling.
        """
        if source_frame_count > 1 and output_frame_count > 1:
            return (
                source_fps
                * (output_frame_count - 1)
                / (source_frame_count - 1)
            )

        return source_fps

    def frame_to_bgr(
        frame: Any,
        *,
        input_color_order: str = "rgb",
    ) -> np.ndarray:
        frame = _to_numpy(frame)

        # Convert channel-first C×H×W to H×W×C.
        if (
            frame.ndim == 3
            and frame.shape[0] in {1, 3, 4}
            and frame.shape[-1] not in {1, 3, 4}
        ):
            frame = np.transpose(frame, (1, 2, 0))

        if (
            frame.ndim != 3
            or frame.shape[-1] not in {1, 3, 4}
        ):
            raise ValueError(
                "Frames must be H×W×C or C×H×W with "
                f"1, 3, or 4 channels; received {frame.shape}."
            )

        if np.issubdtype(frame.dtype, np.floating):
            frame = np.nan_to_num(frame)

            if (
                frame.size
                and frame.min() >= 0
                and frame.max() <= 1
            ):
                frame = frame * 255

        frame = np.clip(frame, 0, 255).astype(np.uint8)

        order = input_color_order.lower()

        if order not in {"rgb", "bgr"}:
            raise ValueError(
                "color_order values must be 'rgb' or 'bgr'."
            )

        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_GRAY2BGR,
            )
        elif frame.shape[-1] == 4:
            frame = cv2.cvtColor(
                frame,
                (
                    cv2.COLOR_RGBA2BGR
                    if order == "rgb"
                    else cv2.COLOR_BGRA2BGR
                ),
            )
        elif order == "rgb":
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )

        return np.ascontiguousarray(frame)

    def encode_mp4(
        *,
        frames: Sequence[Any],
        output_path: Path,
        output_fps: float,
        input_color_order: str = "rgb",
    ) -> None:
        if len(frames) == 0:
            raise ValueError("Cannot encode an empty video.")

        first_frame = frame_to_bgr(
            frames[0],
            input_color_order=input_color_order,
        )
        height, width = first_frame.shape[:2]

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(output_fps),
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Unable to initialize MP4 encoder for {output_path}."
            )

        try:
            writer.write(first_frame)

            for frame_index, frame in enumerate(
                frames[1:],
                start=1,
            ):
                converted = frame_to_bgr(
                    frame,
                    input_color_order=input_color_order,
                )

                if converted.shape[:2] != (height, width):
                    raise ValueError(
                        f"Frame {frame_index} has dimensions "
                        f"{converted.shape[:2]}; expected "
                        f"{(height, width)}."
                    )

                writer.write(converted)
        finally:
            writer.release()

        if (
            not output_path.is_file()
            or output_path.stat().st_size == 0
        ):
            raise RuntimeError(
                f"Encoding produced an empty file: {output_path}"
            )

    with zipfile.ZipFile(
        bundle_path,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        #
        # Existing compressed video paths
        #
        if video_paths_views:
            path_views = [
                _as_path_list(value)
                for value in video_paths_views
            ]

            roles = _roles(
                view_type,
                len(path_views),
            )
            stitch_order = _view_stitch_order(
                roles,
                len(path_views),
            )

            path_views = [
                path_views[index]
                for index in stitch_order
            ]
            roles = [
                roles[index]
                for index in stitch_order
            ]

            if verbose:
                print(
                    "Video-path views will be stitched left-to-right as: "
                    + ", ".join(roles),
                    flush=True,
                )

            episode_count = len(path_views[0])

            if any(
                len(paths) != episode_count
                for paths in path_views
            ):
                raise ValueError(
                    "Every video_paths_viewN must contain the "
                    "same number of episodes."
                )

            validate_episode_count(episode_count)

            for episode_position in range(episode_count):
                view_info = [
                    _video_path_info(paths[episode_position])
                    for paths in path_views
                ]
                frame_counts = [
                    frame_count
                    for frame_count, _source_fps in view_info
                ]

                if len(set(frame_counts)) != 1:
                    raise ValueError(
                        f"Episode {episode_position} views have "
                        f"different frame counts: {frame_counts}"
                    )

                source_frame_count = frame_counts[0]

                if fps is not None:
                    source_fps = float(fps)
                else:
                    source_fps = next(
                        (
                            detected_fps
                            for _frame_count, detected_fps in view_info
                            if detected_fps is not None
                        ),
                        30.0,
                    )

                if (
                    not math.isfinite(source_fps)
                    or source_fps <= 0
                ):
                    raise ValueError(
                        f"Invalid fps for episode {episode_position}: "
                        f"{source_fps!r}"
                    )

                sampled_indices: list[int] | None = None

                if (
                    max_frames is not None
                    and source_frame_count > max_frames
                ):
                    sampled_indices = _uniform_frame_indices(
                        source_frame_count,
                        max_frames,
                    )
                    output_fps = _sampled_fps(
                        source_fps,
                        source_frame_count,
                        len(sampled_indices),
                    )
                else:
                    output_fps = source_fps

                sampling: dict[str, Any] = {
                    "source_frame_count": source_frame_count,
                }

                if sampled_indices is not None:
                    sampling["source_frame_indices"] = sampled_indices

                episode_manifest: dict[str, Any] = {
                    "episode_index": episode_position,
                    "fps": output_fps,
                    "sampling": sampling,
                    "views": [],
                }

                for view_position, (role, paths) in enumerate(
                    zip(roles, path_views)
                ):
                    source = paths[episode_position]
                    local_video_path: Path | None = None

                    try:
                        if sampled_indices is None:
                            payload_path = source
                            suffix = source.suffix.lower() or ".mp4"
                        else:
                            sampled_frames = _read_video_frames_at_indices(
                                source,
                                sampled_indices,
                            )
                            local_video_path = (
                                bundle_path.parent
                                / (
                                    f"episode_{episode_position}_"
                                    f"view_{view_position}_sampled.mp4"
                                )
                            )

                            encode_mp4(
                                frames=sampled_frames,
                                output_path=local_video_path,
                                output_fps=output_fps,
                                input_color_order="bgr",
                            )

                            payload_path = local_video_path
                            suffix = ".mp4"

                        member = (
                            f"episodes/{episode_position}/"
                            f"view_{view_position}{suffix}"
                        )
                        archive.write(payload_path, member)

                        episode_manifest["views"].append(
                            {
                                "role": role,
                                "kind": "video",
                                "path": member,
                            }
                        )

                        if verbose:
                            prepared_count = (
                                len(sampled_indices)
                                if sampled_indices is not None
                                else source_frame_count
                            )
                            print(
                                f"Packaged episode {episode_position}, "
                                f"view {role}: "
                                f"{prepared_count}/{source_frame_count} "
                                "frames.",
                                flush=True,
                            )
                    finally:
                        if local_video_path is not None:
                            local_video_path.unlink(missing_ok=True)

                manifest["episodes"].append(episode_manifest)
        #
        # In-memory video frames
        #
        elif video_frames_views:
            manifest["max_frames"] = max_frames

            frame_views = [
                _split_frame_episodes(
                    value,
                    input_name=f"video_frames_view{index + 1}",
                    verbose=verbose,
                )
                for index, value in enumerate(video_frames_views)
            ]

            episode_count = len(frame_views[0])

            if any(
                len(view) != episode_count
                for view in frame_views
            ):
                raise ValueError(
                    "Every video_frames_viewN must contain the "
                    "same number of episodes."
                )
            
            validate_episode_count(episode_count)

            roles = _roles(
                view_type,
                len(frame_views),
            )

            if isinstance(color_order, str):
                color_orders = [
                    color_order.lower()
                ] * len(frame_views)
            else:
                color_orders = [
                    str(value).lower()
                    for value in color_order
                ]

            if (
                len(color_orders) != len(frame_views)
                or any(
                    value not in {"rgb", "bgr"}
                    for value in color_orders
                )
            ):
                raise ValueError(
                    "color_order must be 'rgb'/'bgr' or "
                    "one value per view."
                )

            stitch_order = _view_stitch_order(
                roles,
                len(frame_views),
            )

            frame_views = [
                frame_views[index]
                for index in stitch_order
            ]
            roles = [
                roles[index]
                for index in stitch_order
            ]
            color_orders = [
                color_orders[index]
                for index in stitch_order
            ]

            if verbose:
                print(
                    "Frame views will be stitched left-to-right as: "
                    + ", ".join(roles),
                    flush=True,
                )

            source_fps = float(fps or 30.0)

            for episode_position in range(episode_count):
                frame_counts = [
                    len(view[episode_position])
                    for view in frame_views
                ]

                if len(set(frame_counts)) != 1:
                    raise ValueError(
                        f"Episode {episode_position} views have "
                        f"different frame counts: {frame_counts}"
                    )

                source_frame_count = frame_counts[0]
                offsets = sampled_offsets(source_frame_count)
                output_fps = sampled_fps(
                    source_fps,
                    source_frame_count,
                    len(offsets),
                )

                sampling: dict[str, Any] = {
                    "source_frame_count": source_frame_count,
                }

                if len(offsets) != source_frame_count:
                    sampling["source_frame_indices"] = [
                        int(offset)
                        for offset in offsets
                    ]

                episode_manifest = {
                    "episode_index": episode_position,
                    "fps": output_fps,
                    "sampling": sampling,
                    "views": [],
                }

                for view_position, (
                    role,
                    views,
                    order,
                ) in enumerate(
                    zip(
                        roles,
                        frame_views,
                        color_orders,
                    )
                ):
                    episode_frames = views[episode_position]
                    selected_frames = [
                        episode_frames[int(offset)]
                        for offset in offsets
                    ]

                    local_video_path = (
                        bundle_path.parent
                        / (
                            f"episode_{episode_position}_"
                            f"view_{view_position}.mp4"
                        )
                    )

                    encode_mp4(
                        frames=selected_frames,
                        output_path=local_video_path,
                        output_fps=output_fps,
                        input_color_order=order,
                    )

                    member = (
                        f"episodes/{episode_position}/"
                        f"view_{view_position}.mp4"
                    )
                    archive.write(local_video_path, member)

                    episode_manifest["views"].append(
                        {
                            "role": role,
                            "kind": "video",
                            "path": member,
                        }
                    )
                    if verbose:
                        print(
                            f"Encoded episode {episode_position}, "
                            f"view {role}: "
                            f"{len(offsets)}/{source_frame_count} "
                            f"frames, "
                            f"{local_video_path.stat().st_size / 1024**2:.1f} MiB",
                            flush=True,
                        )

                    # The MP4 is now inside the .slb bundle.
                    local_video_path.unlink(missing_ok=True)

                manifest["episodes"].append(
                    episode_manifest
                )

            manifest["fps"] = source_fps

        #
        # Local video-backed LeRobotDataset
        #
        else:
            manifest["max_frames"] = max_frames
            (
                lerobot_episode_manifests,
                source_fps,
            ) = _write_lerobot_videos_into_archive(
                archive=archive,
                bundle_path=bundle_path,
                dataset=lerobot_dataset,
                episodes=episodes,
                view_map=lerobot_view_map,
                requested_fps=fps,
                max_frames=max_frames,
                max_videos=max_videos,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                remux_fallback=lerobot_remux_fallback,
                verbose=verbose,
            )
            manifest["episodes"].extend(
                lerobot_episode_manifests
            )
            manifest["fps"] = source_fps

        validate_episode_count(len(manifest["episodes"]))

        archive.writestr(
            "manifest.json",
            json.dumps(
                manifest,
                separators=(",", ":"),
            ),
        )
    if verbose:
        print(
            f"Bundle created: "
            f"{bundle_path.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )

    return len(manifest["episodes"])



class _UploadProgress:
    """File wrapper that displays throttled upload progress."""

    def __init__(self, stream: BinaryIO, *, verbose: bool):
        self._stream = stream
        self._verbose = verbose

        current = stream.tell()
        stream.seek(0, 2)
        self._total = stream.tell()
        stream.seek(current)

        self._uploaded = current
        self._last_render = 0.0

        if self._verbose:
            self._render(force=True)

    def __len__(self) -> int:
        # Allows requests to preserve Content-Length.
        return self._total

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)

        if chunk:
            self._uploaded += len(chunk)

            if self._verbose:
                self._render()

        return chunk

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()

        if (
            not force
            and self._uploaded < self._total
            and now - self._last_render < 0.1
        ):
            return

        self._last_render = now
        ratio = (
            min(self._uploaded / self._total, 1.0)
            if self._total
            else 1.0
        )

        width = 24
        completed = round(width * ratio)
        bar = "#" * completed + "-" * (width - completed)

        print(
            f"\rUploading [{bar}] {ratio:6.1%} "
            f"({self._uploaded / 1024**2:.1f}/"
            f"{self._total / 1024**2:.1f} MiB)",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def finish(self) -> None:
        if self._verbose:
            self._render(force=True)
            print(file=sys.stderr, flush=True)



class Sublinear:
    def __init__(
        self,
        api_key: str,
        base_url: str = SUPABASE_FUNCTIONS_URL,
        **_: Any,
    ):
        if not api_key:
            raise ValueError("api_key cannot be empty")

        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}"}
        )
        self.responses = self

    def cancel_job(self, job_id: str) -> None:
        response = self.session.post(
            f"{self.base_url}/create-inference-job",
            json={"action": "cancel", "job_id": job_id},
            timeout=10,
        )
        if response.status_code != 409:
            response.raise_for_status()


    def create_job(
        self,
        input_json: dict[str, Any],
        *,
        bundle: BinaryIO,
        model: str,
        verbose: bool = False,
    ) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/create-inference-job",
            json={
                "action": "create",
                "filename": f"input-{time.time_ns()}.slb",
                "input_json": input_json,
                "content_type": "application/zip",
                "model": model.lower(),
            },
            timeout=30,
        )
        response.raise_for_status()
        job = response.json()

        upload = _UploadProgress(bundle, verbose=verbose)

        try:
            upload_response = requests.put(
                job["upload_url"],
                data=upload,
                headers={"Content-Type": "application/zip"},
                timeout=(30, 1800),
            )
        finally:
            upload.finish()

        if not upload_response.ok:
            raise RuntimeError(
                f"Storage upload failed ({upload_response.status_code}): "
                f"{upload_response.text}"
            )

        start_response = self.session.post(
            f"{self.base_url}/create-inference-job",
            json={"action": "start", "job_id": job["job_id"]},
            timeout=60,
        )
        start_response.raise_for_status()
        return start_response.json()


    def get_job(self, job_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/get-inference-job",
            params={"job_id": job_id},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 6000,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout

        while True:
            job = self.get_job(job_id)
            status = job["status"]

            if status == "completed":
                return job
            if status == "failed":
                raise RuntimeError(
                    job.get("error") or "Inference job failed."
                )
            if status == "canceled":
                raise RuntimeError("Inference job was canceled.")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job did not finish within {timeout} seconds."
                )

            time.sleep(poll_interval)

    def create(
        self,
        *,
        model: str,
        task_description: str,
        video_path: str | Path | None = None,
        video_paths: Sequence[str | Path] | None = None,
        video_paths_view1: Any = None,
        video_paths_view2: Any = None,
        video_paths_view3: Any = None,
        video_frames: Any = None,
        video_frames_view1: Any = None,
        video_frames_view2: Any = None,
        video_frames_view3: Any = None,
        view_type: str | Sequence[str] | tuple[str, ...] | None = None,
        lerobot_dataset: Any = None,
        episodes: int | Sequence[int] | None = None,
        lerobot_view_map: dict[str, str] | None = None,
        color_order: str | Sequence[str] = "rgb",
        fps: float | None = None,
        max_frames: int | None = 50,
        max_videos: int = _DEFAULT_MAX_VIDEOS_PER_JOB,
        ffmpeg_path: str | Path | None = None,
        ffprobe_path: str | Path | None = None,
        lerobot_remux_fallback: str = "transcode",
        verbose: bool = False,
        **input_json: Any,
    ) -> SimpleNamespace:

        if not task_description.strip():
            raise ValueError("task_description must be provided")

        path_view1 = (
            video_paths_view1
            if video_paths_view1 is not None
            else video_paths
            if video_paths is not None
            else video_path
        )
        frame_view1 = (
            video_frames_view1
            if video_frames_view1 is not None
            else video_frames
        )

        path_views = [
            value
            for value in [
                path_view1,
                video_paths_view2,
                video_paths_view3,
            ]
            if value is not None
        ]
        frame_views = [
            value
            for value in [
                frame_view1,
                video_frames_view2,
                video_frames_view3,
            ]
            if value is not None
        ]

        with tempfile.TemporaryDirectory() as temp_directory:
            bundle_path = Path(temp_directory) / "input.slb"
            episode_count = _write_bundle(
                bundle_path,
                video_paths_views=path_views,
                video_frames_views=frame_views,
                view_type=view_type,
                color_order=color_order,
                lerobot_dataset=lerobot_dataset,
                episodes=episodes,
                lerobot_view_map=lerobot_view_map,
                fps=fps,
                max_frames=max_frames,
                max_videos=max_videos,
                ffmpeg_path=ffmpeg_path,
                ffprobe_path=ffprobe_path,
                lerobot_remux_fallback=lerobot_remux_fallback,
                verbose=verbose,
            )

            job_input = dict(input_json)
            job_input.update(
                {
                    "task_description": task_description,
                    "downsample_to": int(
                        input_json.get("downsample_to", 30)
                    ),
                    "episode_count": episode_count,
                    "bundle_version": 1,
                }
            )

            with bundle_path.open("rb") as bundle:
                created = self.create_job(
                    job_input,
                    bundle=bundle,
                    model=model,
                    verbose=verbose,
                )

            job_id = created["job_id"]

            try:
                finished = self.wait_for_job(job_id)
            except KeyboardInterrupt:
                self.cancel_job(job_id)
                raise

        result = finished.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Job returned an invalid inference result.")

        return SimpleNamespace(**result)





from pathlib import Path

import cv2
import numpy as np

def to_opencv_frame(frame):
    # Handle PyTorch tensors.
    if hasattr(frame, "detach"):
        frame = frame.detach()
    if hasattr(frame, "cpu"):
        frame = frame.cpu()
    if hasattr(frame, "numpy"):
        frame = frame.numpy()

    frame = np.asarray(frame)

    # Convert CHW to HWC.
    if (
        frame.ndim == 3
        and frame.shape[0] in (1, 3, 4)
        and frame.shape[-1] not in (1, 3, 4)
    ):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim != 3 or frame.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    # Convert float [0, 1] or other numeric input to uint8.
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.nan_to_num(frame)

        if frame.size and frame.min() >= 0 and frame.max() <= 1:
            frame = frame * 255

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    # LeRobot frames are RGB; OpenCV writes BGR.
    if frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return np.ascontiguousarray(frame)


def write_mp4(frames, output_path, fps=30.0):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_first = to_opencv_frame(frames[0])
    height, width = converted_first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height), 
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create {output_path}")

    try:
        writer.write(converted_first)

        for frame in frames[1:]:
            converted = to_opencv_frame(frame)

            if converted.shape[:2] != (height, width):
                raise ValueError("All frames must have the same dimensions.")

            writer.write(converted)
    finally:
        writer.release()

    return output_path





def load_video_frames(
    video_path: str,
    max_frames: int = 100,
    frame_indices=None,
    return_metadata: bool = False,
    keep_as_bgr: bool = False,
):
    """
    Load at most `max_frames` uniformly sampled frames.
    Returns:
        frames
    Or, when return_metadata=True:
        frames, sampled_indices, total_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(
            f"Could not determine frame count for video: {video_path}"
        )
    if frame_indices is None:
        sampled_indices = np.linspace(
            0,
            total_frames - 1,
            num=min(max_frames, total_frames),
            dtype=int,
        )
    else:
        sampled_indices = np.asarray(frame_indices, dtype=int)
        if np.any(sampled_indices < 0) or np.any(
            sampled_indices >= total_frames
        ):
            cap.release()
            raise ValueError(
                f"Requested frame index is outside video range "
                f"[0, {total_frames - 1}]"
            )
    frames = []
    valid_indices = []
    for index in sampled_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ret, frame = cap.read()
        if not ret:
            continue
        if not keep_as_bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        valid_indices.append(int(index))
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be decoded from: {video_path}")
    if return_metadata:
        return frames, valid_indices, total_frames
    return frames





def resize_and_pad_to_target(
    frame,
    target_height=768,
    min_width=768,
    max_width=1536,
    pad_color=(255, 255, 255),
):
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")
    h, w = frame.shape[:2]
    # Scale so height is exactly target_height.
    scale = target_height / h
    new_w = int(round(w * scale))
    frame = cv2.resize(
        frame,
        (new_w, target_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    h, w = frame.shape[:2]
    # Clamp width while preserving aspect ratio, then pad vertically.
    if w > max_width:
        scale = max_width / w
        new_h = int(round(h * scale))
        frame = cv2.resize(
            frame,
            (max_width, new_h),
            interpolation=cv2.INTER_AREA,
        )
        h, w = frame.shape[:2]
        pad_top = max((target_height - h) // 2, 0)
        pad_bottom = max(target_height - h - pad_top, 0)
        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            0,
            0,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )
        h, w = frame.shape[:2]
    # Pad narrow frames.
    pad_left = max((min_width - w) // 2, 0)
    pad_right = max(min_width - w - pad_left, 0)
    if pad_left or pad_right:
        frame = cv2.copyMakeBorder(
            frame,
            0,
            0,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )
    return frame



import re

def classify_view_type(value):
    if value is None:
        return None

    tokens = {
        token
        for token in re.sub(
            r"[^a-z0-9]+",
            "_",
            str(value).strip().lower(),
        ).split("_")
        if token
    }

    is_static = bool(
        tokens.intersection({"top", "static", "external", "side", "front"})
    )
    is_wrist = "wrist" in tokens
    is_left = "left" in tokens
    is_right = "right" in tokens

    # A label containing both categories is ambiguous.
    if is_static and is_wrist:
        return None

    if is_wrist:
        if is_left and is_right:
            return None
        if is_left:
            return "left_wrist"
        if is_right:
            return "right_wrist"
        return "wrist"

    if is_static:
        return "static"

    return None


def view_stitch_order(types, view_count):
    default_order = list(range(view_count))

    if types is None or len(types) != view_count:
        return default_order

    categories = [classify_view_type(value) for value in types]

    if view_count == 3:
        desired = ["left_wrist", "static", "right_wrist"]

        if set(categories) == set(desired):
            return [categories.index(category) for category in desired]

    elif view_count == 2:
        simplified = [
            "wrist"
            if category in {"left_wrist", "right_wrist", "wrist"}
            else category
            for category in categories
        ]
        desired = ["static", "wrist"]

        if set(simplified) == set(desired):
            return [simplified.index(category) for category in desired]

    # Unknown, ambiguous, or duplicate categories: preserve view1, view2, ...
    return default_order


def _is_torch_tensor(value: Any) -> bool:
    """
    Detect a PyTorch tensor without making torch a required dependency.
    """
    return (
        value.__class__.__module__.split(".", 1)[0] == "torch"
        and hasattr(value, "detach")
        and hasattr(value, "cpu")
        and hasattr(value, "numpy")
    )


def _normalize_video_plot_frames(
    value: Any,
    *,
    input_name: str,
    verbose: bool = False,
) -> np.ndarray:
    """
    Normalize one video to T×H×W×3.

    Accepted inputs:
        NumPy array or PyTorch tensor with shape:
          T×C×H×W
          T×H×W×C
          1×T×C×H×W
          1×T×H×W×C

    Five-dimensional input is accepted only when B == 1.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            raise ValueError(
                f"{input_name} cannot be an empty frame sequence."
            )

        frame_arrays: list[np.ndarray] = []

        for frame_index, frame in enumerate(value):
            if (
                not isinstance(frame, np.ndarray)
                and not _is_torch_tensor(frame)
            ):
                raise TypeError(
                    f"{input_name}[{frame_index}] must be a NumPy "
                    "array or PyTorch tensor; received "
                    f"{type(frame).__name__}."
                )

            frame_array = _to_numpy(frame)

            if frame_array.ndim != 3:
                raise ValueError(
                    f"{input_name}[{frame_index}] must have three "
                    "dimensions, either H×W×C or C×H×W; received "
                    f"shape {frame_array.shape}."
                )

            frame_arrays.append(frame_array)

        try:
            # A sequence of T individual frames becomes one video:
            #   H×W×C frames -> T×H×W×C
            #   C×H×W frames -> T×C×H×W
            array = np.stack(frame_arrays, axis=0)
        except ValueError as error:
            frame_shapes = [
                frame.shape
                for frame in frame_arrays
            ]
            raise ValueError(
                f"All frames in {input_name} must have the same "
                f"shape; received {frame_shapes}."
            ) from error

        if verbose:
            print(
                f"{input_name}: stacked {len(frame_arrays)} frames "
                f"into shape {array.shape}.",
                flush=True,
            )

    elif isinstance(value, np.ndarray) or _is_torch_tensor(value):
        array = _to_numpy(value)

    else:
        raise TypeError(
            f"{input_name} must be a NumPy array, PyTorch tensor, "
            "or a list/tuple of frame arrays; received "
            f"{type(value).__name__}."
        )

    if array.ndim not in (4, 5):
        raise ValueError(
            f"{input_name} must have 4 or 5 dimensions; "
            f"received shape {array.shape}."
        )

    if array.ndim == 5:
        batch_size = array.shape[0]

        if batch_size != 1:
            raise ValueError(
                f"{input_name} contains B={batch_size} videos, but "
                "video_plot supports only one video. Pass a 4D video "
                "or a 5D batch with B=1."
            )

        channel_first = array.shape[2] == 3
        channel_last = array.shape[-1] == 3

        if channel_first == channel_last:
            raise ValueError(
                f"{input_name} has an ambiguous or missing RGB "
                f"channel in shape {array.shape}. Expected "
                "(1, T, C, H, W) or (1, T, H, W, C), with C=3."
            )

        if channel_first:
            # 1×T×C×H×W -> T×H×W×C
            normalized = np.moveaxis(array[0], 1, -1)
            layout = "(1, T, C, H, W)"
        else:
            # 1×T×H×W×C -> T×H×W×C
            normalized = array[0]
            layout = "(1, T, H, W, C)"

        if verbose:
            print(
                f"{input_name}: received shape {array.shape}; "
                f"assuming a one-video batch in {layout} layout "
                f"with T={array.shape[1]}.",
                flush=True,
            )

    else:
        channel_first = array.shape[1] == 3
        channel_last = array.shape[-1] == 3

        if channel_first == channel_last:
            raise ValueError(
                f"{input_name} has an ambiguous or missing RGB "
                f"channel in shape {array.shape}. Expected "
                "(T, C, H, W) or (T, H, W, C), with C=3."
            )

        if channel_first:
            # T×C×H×W -> T×H×W×C
            normalized = np.moveaxis(array, 1, -1)
            layout = "(T, C, H, W)"
        else:
            normalized = array
            layout = "(T, H, W, C)"

        if verbose:
            print(
                f"{input_name}: received shape {array.shape}; "
                f"assuming one video in {layout} layout "
                f"with T={array.shape[0]}.",
                flush=True,
            )

    if len(normalized) == 0:
        raise ValueError(
            f"{input_name} cannot contain zero frames."
        )

    return np.ascontiguousarray(normalized)

def load_plot_source(
    path,
    frames,
    frame_limit,
    *,
    input_name: str = "video_frames",
    verbose: bool = False,
):
    if path is not None and frames is not None:
        raise ValueError(
            "Provide either a path or frames for each view, not both."
        )

    if path is not None:
        loaded_frames, indices, original_count = load_video_frames(
            path,
            max_frames=frame_limit,
            return_metadata=True,
        )

        return (
            [
                _frame_to_rgb_uint8(frame)
                for frame in loaded_frames
            ],
            [int(index) for index in indices],
            int(original_count),
        )

    if frames is not None:
        video = _normalize_video_plot_frames(
            frames,
            input_name=input_name,
            verbose=verbose,
        )
        original_count = len(video)

        output_count = min(
            frame_limit,
            original_count,
        )
        indices = np.linspace(
            0,
            original_count - 1,
            num=output_count,
            dtype=int,
        ).tolist()

        return (
            [
                _frame_to_rgb_uint8(video[index])
                for index in indices
            ],
            indices,
            original_count,
        )

    raise ValueError("Missing video source.")

def load_lerobot_plot_sources(
    dataset,
    episode_index,
    view_map,
    frame_limit,
):
    if episode_index is None:
        raise ValueError(
            "episode is required when lerobot_dataset is provided."
        )

    selected = _selected_lerobot_episode_indices(
        dataset,
        episode_index,
        limit=1,
    )
    episode_index = selected[0]

    meta = dataset.meta
    video_keys = _lerobot_video_keys(meta)
    depth_keys = set(getattr(meta, "depth_keys", ()) or ())

    if view_map is None:
        defaults = {
            "static": "observation.images.top",
            "left_wrist": "observation.images.left_wrist",
            "right_wrist": "observation.images.right_wrist",
        }
        view_map = {
            role: camera_key
            for role, camera_key in defaults.items()
            if camera_key in video_keys
        }
    else:
        view_map = dict(view_map)

    if not view_map:
        raise ValueError(
            "No video-backed LeRobot views were found. "
            "Provide lerobot_view_map."
        )

    if any(not str(role).strip() for role in view_map):
        raise ValueError(
            "LeRobot view roles must be non-empty strings."
        )
    view_map = _ordered_lerobot_view_map(view_map)

    unavailable = [
        camera_key
        for camera_key in view_map.values()
        if camera_key not in video_keys
    ]
    if unavailable:
        raise ValueError(
            "These LeRobot features are not video-backed or do not exist: "
            f"{unavailable}"
        )

    requested_depth = [
        camera_key
        for camera_key in view_map.values()
        if camera_key in depth_keys
    ]
    if requested_depth:
        raise ValueError(
            f"Depth-video features cannot be plotted as RGB: "
            f"{requested_depth}"
        )

    if getattr(dataset, "image_transforms", None) is not None:
        raise ValueError(
            "video_plot cannot apply LeRobot image_transforms when "
            "reading the compressed video files directly."
        )

    meta_fps = getattr(meta, "fps", None)
    if meta_fps is None:
        meta_fps = getattr(dataset, "fps", None)
    if meta_fps is None:
        raise ValueError("LeRobot fps metadata is unavailable.")

    source_fps = float(meta_fps)
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"Invalid LeRobot fps: {source_fps!r}")

    row = _episode_metadata(meta, episode_index)
    dataset_root = Path(dataset.root)

    view_specs = []
    expected_frame_count = None

    for role, camera_key in view_map.items():
        (
            start,
            _stop,
            view_frame_count,
            _explicit_bounds,
        ) = _episode_clip_spec(
            row,
            camera_key,
            source_fps,
        )

        if expected_frame_count is None:
            expected_frame_count = view_frame_count
        elif view_frame_count != expected_frame_count:
            raise ValueError(
                f"Episode {episode_index} views have different frame "
                f"counts: {expected_frame_count} and {view_frame_count}."
            )

        relative_path = Path(
            meta.get_video_file_path(
                episode_index,
                camera_key,
            )
        )
        source_path = (
            relative_path
            if relative_path.is_absolute()
            else dataset_root / relative_path
        ).resolve()

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        view_specs.append(
            {
                "role": str(role).strip(),
                "camera_key": camera_key,
                "source_path": source_path,
                "start_frame": round(start * source_fps),
            }
        )

    if frame_limit is None:
        frame_limit = expected_frame_count

    if (
        isinstance(frame_limit, bool)
        or not isinstance(frame_limit, int)
        or frame_limit < 1
    ):
        raise ValueError("max_frames must be a positive integer or None.")

    sampled_indices = _uniform_frame_indices(
        expected_frame_count,
        frame_limit,
    )

    loaded_sources = []

    for spec in view_specs:
        capture = cv2.VideoCapture(str(spec["source_path"]))

        if not capture.isOpened():
            raise ValueError(
                f"Unable to open LeRobot video: "
                f"{spec['source_path']}"
            )

        frames = []

        try:
            for relative_index in sampled_indices:
                absolute_index = (
                    spec["start_frame"] + relative_index
                )

                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    absolute_index,
                )
                ok, frame = capture.read()

                if not ok:
                    raise ValueError(
                        "Unable to decode LeRobot frame "
                        f"{relative_index} for episode {episode_index}, "
                        f"view {spec['camera_key']}."
                    )

                # video_plot operates on RGB frames.
                frames.append(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )
                )
        finally:
            capture.release()

        loaded_sources.append(
            (
                frames,
                list(sampled_indices),
                expected_frame_count,
            )
        )

    return loaded_sources, [
        spec["role"]
        for spec in view_specs
    ]


def _unwrap_singleton_output_batch(
    value: Any,
    *,
    field_name: str,
) -> Any:
    if value is None:
        return None

    try:
        array = _to_numpy(value)
    except (TypeError, ValueError):
        array = None

    if (
        array is not None
        and array.dtype != object
        and array.ndim >= 2
    ):
        if array.shape[0] != 1:
            raise ValueError(
                f"{field_name} contains {array.shape[0]} videos, "
                "but video_plot supports only one."
            )

        return array[0].tolist()

    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) > 0
        and (
            isinstance(value[0], Sequence)
            or hasattr(value[0], "shape")
        )
    ):
        if len(value) != 1:
            raise ValueError(
                f"{field_name} contains {len(value)} videos, "
                "but video_plot supports only one."
            )

        return _to_numpy(value[0]).tolist()

    return value

def video_plot(outputs, plot_save_path: str, 
                video_path=None, 
                video_path_view1: str = None, 
                video_path_view2: str = None,
                video_path_view3: str = None,
                video_frames: Any = None,
                video_frames_view1: Any = None,
                video_frames_view2: Any = None,
                video_frames_view3: Any = None,
                view_type: str | Sequence[str] | tuple[str, ...] | None = None,
                lerobot_dataset=None,
                episode: int | None = None,
                lerobot_view_map: dict[str, str] | None = None,
                max_frames: int | None = 50,
                task_description: str =None, 
                show_all_frames: bool = False,
                fps=2, wrap_width=26, font_scale=1, font_height=30, text_thickness=2, line_type=2, show_output_text=True, cfg=None, env_rew_lab='Ground-truth reward', 
                save_json = True, verbose=False, write_json = True, x_axis_label='Timestep', y_axis_label='Predicted task progress (%)', return_frames=False, 
                target_video_height: int = 768, plot_line_width: int = 5, show_legend=True, legend_font_size: int |str = None, 
                x_axis_label_size: int = None, y_axis_label_size: int = None, x_axis_tick_size: int = None, y_axis_tick_size: int = None, plot_width_relative_to_video: float = 1/2):
    # 
    ############ EXTRACT VIDEO FRAMES FOR ALL VIDEOS AS A LIST OF LISTS    
    all_inputs = (
        video_path,
        video_path_view1,
        video_path_view2,
        video_path_view3,
        video_frames,
        video_frames_view1,
        video_frames_view2,
        video_frames_view3,
    )

    outputs = copy.deepcopy(
        [
            output
            for output in outputs
            if output is not None
        ]
    )

    if not outputs:
        raise ValueError("outputs cannot be empty.")

    for output in outputs:
        for field_name in (
            "rewards",
            "output_text",
            "success_probs",
        ):
            if field_name in output:
                output[field_name] = _unwrap_singleton_output_batch(
                    output[field_name],
                    field_name=field_name,
                )

    if (
        lerobot_dataset is None
        and all(value is None for value in all_inputs)
    ):
        raise ValueError(
            "Provide video paths, video frames, or lerobot_dataset."
        )

    reward_len_list = []

    for output in outputs:
        if "rewards" not in output:
            raise ValueError(
                "Each output must contain a 'rewards' key."
            )

        reward_len_list.append(
            len(output["rewards"])
        )

    if len(set(reward_len_list)) != 1:
        raise ValueError(
            "Length of rewards is inconsistent across outputs: "
            f"{reward_len_list}"
        )

    if max_frames is None:
        frame_limit = reward_len_list[0]
    elif (
        isinstance(max_frames, bool)
        or not isinstance(max_frames, int)
        or max_frames < 1
    ):
        raise ValueError(
            "max_frames must be a positive integer or None."
        )
    else:
        frame_limit = min(
            max_frames,
            reward_len_list[0],
        )

    if isinstance(fps, bool):
        raise ValueError("fps must be a positive number.")

    output_fps = float(fps)

    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError(
            "fps must be a positive finite number."
        )

    generic_source_present = (
        video_path is not None
        or video_frames is not None
    )

    numbered_source_specs = [
        (
            "video_frames_view1",
            video_path_view1,
            video_frames_view1,
        ),
        (
            "video_frames_view2",
            video_path_view2,
            video_frames_view2,
        ),
        (
            "video_frames_view3",
            video_path_view3,
            video_frames_view3,
        ),
    ]
    numbered_source_specs = [
        source
        for source in numbered_source_specs
        if source[1] is not None or source[2] is not None
    ]

    input_family_count = sum(
        (
            generic_source_present,
            bool(numbered_source_specs),
            lerobot_dataset is not None,
        )
    )

    if input_family_count != 1:
        raise ValueError(
            "Provide exactly one of video_path/video_frames, "
            "numbered views, or lerobot_dataset."
        )

    if lerobot_dataset is not None:
        (
            loaded_sources,
            effective_view_types,
        ) = load_lerobot_plot_sources(
            dataset=lerobot_dataset,
            episode_index=episode,
            view_map=lerobot_view_map,
            frame_limit=frame_limit,
        )

    else:
        if generic_source_present:
            source_specs = [
                (
                    "video_frames",
                    video_path,
                    video_frames,
                )
            ]
        else:
            source_specs = numbered_source_specs

        loaded_sources = [
            load_plot_source(
                path,
                frames,
                frame_limit,
                input_name=input_name,
                verbose=verbose,
            )
            for input_name, path, frames in source_specs
        ]

        effective_view_types = _roles(
            view_type,
            len(loaded_sources),
        )

    original_counts = [
        source[2]
        for source in loaded_sources
    ]
    sampled_indices_per_view = [
        source[1]
        for source in loaded_sources
    ]



    if len(set(original_counts)) != 1:
        raise ValueError(
            "All views must have the same frame count: "
            f"{original_counts}"
        )

    if any(
        indices != sampled_indices_per_view[0]
        for indices in sampled_indices_per_view[1:]
    ):
        raise ValueError(
            "All views must resolve to the same sampled frame indices."
        )

    stitch_order = _view_stitch_order(
        effective_view_types,
        len(loaded_sources),
    )
    ordered_view_types = [
        effective_view_types[index]
        for index in stitch_order
    ]
    ordered_frame_lists = [
        loaded_sources[index][0]
        for index in stitch_order
    ]

    if verbose:
        print(
            "Stitching views left-to-right as: "
            + ", ".join(ordered_view_types),
            flush=True,
        )

    frame_list = _stitch_frame_lists(
        ordered_frame_lists
    )

    orig_frame_list_len = original_counts[0]
    sampled_indices = sampled_indices_per_view[0]
    # 
    # 
    # 
    # Validate that the number of rewards matches the number of frames for each output
    if len(frame_list) != frame_limit:
        if len(frame_list) < frame_limit and orig_frame_list_len < frame_limit:
            frame_limit = orig_frame_list_len
            if verbose:
                print(f"frame_limit is greater than the original frame count. Setting frame_limit to {orig_frame_list_len}")
        else:
            raise ValueError(f"downsampled frame list length {len(frame_list)} is great than frame_limit {frame_limit}")
    # 
    if not reward_len_list[0] == orig_frame_list_len:
        raise ValueError(f"Original length of rewards {reward_len_list[0]} does not match original frame length {orig_frame_list_len}")
    # 
    video_downsample_to_idx = np.asarray(
        sampled_indices,
        dtype=int,
    )
    for output in outputs:
        output["sampled_rewards"] = [
            output["rewards"][index]
            for index in video_downsample_to_idx
        ]
        if output.get("output_text") is not None:
            output["sampled_output_text"] = [
                output["output_text"][index]
                for index in video_downsample_to_idx
            ]
    # 
    if write_json and save_json:
        json_save_path = (
            plot_save_path.rsplit(".", 1)[0] + ".json"
        )
        Path(json_save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        json_dict = {
            "outputs": outputs,
            "meta": {
                "source_frame_count": orig_frame_list_len,
                "sampled_frame_indices": (
                    video_downsample_to_idx.tolist()
                ),
            },
        }
        with open(json_save_path, "w", encoding="utf-8",) as json_file:
            json.dump(json_dict, json_file)
    # 
    # only show reasoning traces in video if only showing one model - if "output_text" in more than 2 outputs, set output_text to None since this is too much information to show in the video
    count_output_text = 0
    output_rewards_len = []
    output_rewards_concat = []
    final_output_text_idx_list = []
    for output in outputs:
        output_rewards_len.append(len(output["sampled_rewards"]))
        output_rewards_concat += output["sampled_rewards"]
        if "sampled_output_text" in output and output["sampled_output_text"] is not None:
            count_output_text += 1
            final_output_text = output["sampled_output_text"]
            final_output_text_idx_list = []
            for step_idx in range(len(final_output_text)):
                if step_idx==0 or not final_output_text[step_idx].strip() == "":
                    final_output_text_idx_list.append(step_idx)
    # 
    if not count_output_text == 1:
        show_output_text = False
    # 
    if show_output_text:
        if not show_all_frames:
            if verbose:
                print('Only showing reasoning traces for frames where reasoning traces are not empty. If you want to show reasoning traces for all frames, set show_all_frames to True.')
            output_rewards_len = []
            output_rewards_concat = []
            for output_idx in range(len(outputs)):
                if "sampled_output_text" in outputs[output_idx] and outputs[output_idx]["sampled_output_text"] is not None:
                    try:
                        outputs[output_idx]["sampled_output_text"] = [outputs[output_idx]["sampled_output_text"][i] for i in final_output_text_idx_list]
                    except:
                        if verbose:
                            print(f"Error in indexing reasoning traces for output_idx {output_idx}. This may be due to a mismatch between the length of reasoning traces and the length of rewards. Setting reasoning traces to None for this output.")
                outputs[output_idx]["sampled_rewards"] = [outputs[output_idx]["sampled_rewards"][i] for i in final_output_text_idx_list]
                output_rewards_len.append(len(outputs[output_idx]["sampled_rewards"]))
                output_rewards_concat += outputs[output_idx]["sampled_rewards"]
            final_output_text = [final_output_text[i] for i in final_output_text_idx_list]
            frame_list = [frame_list[i] for i in final_output_text_idx_list]
    # 
    text_title=''
    if True:
        plt.rcParams.update({'font.size': 12})  # Adjust font size as needed
        #             # 
        frame = frame_list[0]
        # if isinstance(frame, Image.Image):
        #     frame = np.array(frame.convert("RGB"))
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #
        # target_height = frame.shape[0]
        # frame = shape_to_target(frame, target=target_height)
        # target_height = 768
        # target_height = target_video_height
        # # target_height = 400
        # frame = resize_and_pad_to_target(frame, target_height=target_height, min_width=target_height, max_width=target_height*2)
        # frame_height, frame_width = frame.shape[:2]

        target_height = int(target_video_height)

        frame = resize_and_pad_to_target(
            frame,
            target_height=target_height,
            min_width=target_height,
            max_width=target_height * 2,
        )

        reference_height, frame_width = frame.shape[:2]

        if plot_width_relative_to_video <= 0:
            raise ValueError(
                "plot_width_relative_to_video must be greater than zero."
            )

        frame_width -= frame_width % 2

        requested_plot_size = max(
            2,
            round(frame_width * plot_width_relative_to_video),
        )
        requested_plot_size -= requested_plot_size % 2

        if show_output_text:
            # Fixed-height layout so the text column has predictable space.
            frame_height = reference_height
            frame_height -= frame_height % 2

            plot_size = min(
                requested_plot_size,
                frame_height,
            )
            plot_size -= plot_size % 2

            white_box_width = frame_height

        else:
            # Preserve the previous compact, dynamically fitted layout.
            plot_size = requested_plot_size
            frame_height = plot_size
            white_box_width = 0

        top_pad = 50 if task_description is not None else 0

        output_width = (
            frame_width
            + white_box_width
            + plot_size
        )
        output_height = frame_height + top_pad

        print(
            "output_width, output_height:",
            output_width,
            output_height,
        )


        if plot_save_path.endswith(".mp4"):
            plot_save_path = str(
                Path(plot_save_path).with_suffix(".avi")
            )
        Path(plot_save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        # 
        if not os.path.exists(os.path.dirname(plot_save_path)):
            os.makedirs(os.path.dirname(plot_save_path))
        # 
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(plot_save_path, fourcc, output_fps, (frame_width, frame_height))
        out = cv2.VideoWriter(plot_save_path, fourcc, output_fps, (output_width, output_height))
        # 
        fig, ax = plt.subplots(dpi=200)  # Increase DPI for higher resolution
        canvas = FigureCanvas(fig)
        # xdata, ydata, ydata2 = [], [], []
        xdata, ydata, ydata2, ydata3, ydata4, ydata5, ydata6, ydata7 = [], [], [], [], [], [], [], []
        # xlim_max = max(len(predicted_rewards), len(groundtruth_rewards))
        xlim_max = max(output_rewards_len)
        ax.set_xlim(0, xlim_max-1)
        ax.set_title(text_title)
        ymax=100
        # 
        # ax.set_ylim(min(predicted_rewards+groundtruth_rewards), 100)
        ax.set_ylim(min(output_rewards_concat), 100)
        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(x_axis_label)
        if x_axis_label_size is not None:
            ax.xaxis.label.set_size(x_axis_label_size)
        if y_axis_label_size is not None:
            ax.yaxis.label.set_size(y_axis_label_size)
        if x_axis_tick_size is not None:
            ax.tick_params(axis='x', labelsize=x_axis_tick_size)
        if y_axis_tick_size is not None:
            ax.tick_params(axis='y', labelsize=y_axis_tick_size)
        # 
        ln, = ax.plot([], [], 'r', linewidth=plot_line_width,label=outputs[0]["model"])
        # ln.set_color('darkblue')
        # ln.set_markerfacecolor('darkblue')
        # get outputs[0]["plot_color"] if it exists, otherwise use 'darkblue'
        if outputs[0] is not None and "plot_color" in outputs[0]:
            ln.set_color(outputs[0]["plot_color"])
            ln.set_markerfacecolor(outputs[0]["plot_color"])
        else:
            ln.set_color('darkblue')
            ln.set_markerfacecolor('darkblue')
        ln.set_markersize(plot_line_width)
        # 
        ln2, ln3, ln4, ln5, ln6, ln7 = None, None, None, None, None, None
        if len(outputs) > 1:
            ln2, = ax.plot([], [], 'b', linewidth=plot_line_width,label=outputs[1]["model"])
            # ln2.set_color('darkred')
            # ln2.set_markerfacecolor('darkred')
            if outputs[1] is not None and "plot_color" in outputs[1]:
                ln2.set_color(outputs[1]["plot_color"])
                ln2.set_markerfacecolor(outputs[1]["plot_color"])
            else:
                ln2.set_color('darkred')
                ln2.set_markerfacecolor('darkred')
            ln2.set_markersize(plot_line_width)
        if len(outputs) > 2:
            ln3, = ax.plot([], [], 'g', linewidth=plot_line_width,label=outputs[2]["model"])
            # ln3.set_color('darkgreen')
            # ln3.set_markerfacecolor('darkgreen')
            if outputs[2] is not None and "plot_color" in outputs[2]:
                ln3.set_color(outputs[2]["plot_color"])
                ln3.set_markerfacecolor(outputs[2]["plot_color"])
            else:
                ln3.set_color('darkgreen')
                ln3.set_markerfacecolor('darkgreen')
            ln3.set_markersize(plot_line_width)
        if len(outputs) > 3:
            ln4, = ax.plot([], [], 'y', linewidth=plot_line_width,label=outputs[3]["model"])
            # ln4.set_color('orange')
            # ln4.set_markerfacecolor('orange')
            if outputs[3] is not None and "plot_color" in outputs[3]:
                ln4.set_color(outputs[3]["plot_color"])
                ln4.set_markerfacecolor(outputs[3]["plot_color"])
            else:
                ln4.set_color('orange')
                ln4.set_markerfacecolor('orange')
            ln4.set_markersize(plot_line_width)
        if len(outputs) > 4:
            ln5, = ax.plot([], [], 'c', linewidth=plot_line_width,label=outputs[4]["model"])
            # ln5.set_color('cyan')
            # ln5.set_markerfacecolor('cyan')
            if outputs[4] is not None and "plot_color" in outputs[4]:
                ln5.set_color(outputs[4]["plot_color"])
                ln5.set_markerfacecolor(outputs[4]["plot_color"])
            else:
                ln5.set_color('cyan')
                ln5.set_markerfacecolor('cyan')
            ln5.set_markersize(plot_line_width)
        if len(outputs) > 5:
            ln6, = ax.plot([], [], 'm', linewidth=plot_line_width,label=outputs[5]["model"])
            # ln6.set_color('magenta')
            # ln6.set_markerfacecolor('magenta')
            if outputs[5] is not None and "plot_color" in outputs[5]:
                ln6.set_color(outputs[5]["plot_color"])
                ln6.set_markerfacecolor(outputs[5]["plot_color"])
            else:
                ln6.set_color('magenta')
                ln6.set_markerfacecolor('magenta')
            ln6.set_markersize(plot_line_width)
        if len(outputs) > 6:
            ln7, = ax.plot([], [], 'k', linewidth=plot_line_width,label=outputs[6]["model"])
            # ln7.set_color('black')
            # ln7.set_markerfacecolor('black')
            if outputs[6] is not None and "plot_color" in outputs[6]:
                ln7.set_color(outputs[6]["plot_color"])
                ln7.set_markerfacecolor(outputs[6]["plot_color"])
            else:
                ln7.set_color('black')
                ln7.set_markerfacecolor('black')
            ln7.set_markersize(plot_line_width)
        if len(outputs) > 7:
            assert False, f"Plotting more than 7 outputs in the same video is not currently supported."
        # 
        if not show_legend:
            ax.legend_.remove() if ax.legend_ else None
        else:
            if legend_font_size is None:
                if len(outputs) <= 3:
                    ax.legend(fontsize='small')
                elif len(outputs) <= 4:
                    ax.legend(fontsize='x-small')
                else:
                    ax.legend(fontsize='xx-small')
            else:
                ax.legend(fontsize=legend_font_size)
        frame_number = 0
        data_point_number = 0
        while frame_number<xlim_max:
            frame = frame_list[frame_number]
            # if isinstance(frame, Image.Image):
            #     frame = np.array(frame.convert("RGB"))
            #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 
            # frame = shape_to_target(frame, target=target_height)
            # frame = resize_and_pad_to_target(frame, target_height=target_height, min_width=target_height, max_width=target_height*2)
            frame = cv2.resize(
                frame,
                (frame_width, frame_height),
                interpolation=cv2.INTER_AREA
            )
            frame_height, frame_width = frame.shape[:2]
            # fig.set_size_inches(target_height / fig.dpi, target_height / fig.dpi)  
            fig.set_size_inches(plot_size / fig.dpi, plot_size / fig.dpi)
            fig.tight_layout(pad=0.4)  # Adjust layout to ensure everything fits without being cut off
            # 
            if True:
                xdata.append(data_point_number)
                # if len(predicted_rewards) > data_point_number:
                if len(outputs[0]["sampled_rewards"]) > data_point_number:
                    ydata.append(outputs[0]["sampled_rewards"][data_point_number])
                    ln.set_data(xdata, ydata)
                    ax.draw_artist(ln)
                # 
                ax.draw_artist(ax.patch)
                # if len(groundtruth_rewards) > data_point_number:
                if ln2 is not None:
                    ydata2.append(outputs[1]["sampled_rewards"][data_point_number])
                    ln2.set_data(xdata, ydata2)
                    ax.draw_artist(ln2)
                if ln3 is not None:
                    ydata3.append(outputs[2]["sampled_rewards"][data_point_number])
                    ln3.set_data(xdata, ydata3)
                    ax.draw_artist(ln3)
                if ln4 is not None:
                    ydata4.append(outputs[3]["sampled_rewards"][data_point_number])
                    ln4.set_data(xdata, ydata4)
                    ax.draw_artist(ln4)
                if ln5 is not None:
                    ydata5.append(outputs[4]["sampled_rewards"][data_point_number])
                    ln5.set_data(xdata, ydata5)
                    ax.draw_artist(ln5)
                if ln6 is not None:
                    ydata6.append(outputs[5]["sampled_rewards"][data_point_number])
                    ln6.set_data(xdata, ydata6)
                    ax.draw_artist(ln6)
                if ln7 is not None:    
                    ydata7.append(outputs[6]["sampled_rewards"][data_point_number])
                    ln7.set_data(xdata, ydata7)
                    ax.draw_artist(ln7)
                # 
                canvas.draw()
                plot_image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (4,))
            # 
            # 
            # plot_image_resized = cv2.resize(plot_image, (target_height, target_height), interpolation=cv2.INTER_LINEAR)
            plot_image_resized = cv2.resize(
                plot_image,
                (plot_size, plot_size),
                interpolation=cv2.INTER_LINEAR
            )
            # plot_image_rgb = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2RGB)
            # plot_image_bgr = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2BGR)
            plot_image_rgb = cv2.cvtColor(
                plot_image_resized,
                cv2.COLOR_RGBA2RGB,
            )
            # Pad plot vertically to match video height
            pad_total = frame_height - plot_size
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            plot_image_rgb = cv2.copyMakeBorder(
                plot_image_rgb,
                pad_top,
                pad_bottom,
                0,
                0,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
            # 
            if show_output_text:
                # white_box_width = frame_width // denom_  # Adjust width of the white box
                # white_box_width = target_height  # Set white box width to match the height of the frame
                white_box = np.full(
                    (frame_height, white_box_width, 3),
                    255,
                    dtype=np.uint8,
                )
                white_box = np.ones((frame_height, white_box_width, 3), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_color = (0, 0, 0)  
                thickness = text_thickness
                line_type = line_type
                text = final_output_text[frame_number]
                language_description_split = text.split(' ')
                language_description_wrapped = ''
                line_i_len = 0
                for i in range(len(language_description_split)):
                    if line_i_len < wrap_width:
                        language_description_wrapped += language_description_split[i] + ' '
                        line_i_len += len(language_description_split[i])
                    else:
                        language_description_wrapped += '\n' + language_description_split[i] + ' '
                        line_i_len = len(language_description_split[i])
                text = language_description_wrapped
                text = text.replace('</think><answer>', '</think>\n<answer>')
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (white_box_width - text_size[0]) // 2
                text_y = (frame_height + text_size[1]) // 2
                lines = text.split('\n')
                start_y = font_height 
                for i, line in enumerate(lines):
                    y = start_y + (i * font_height)  
                    res_ = cv2.putText(white_box, line, (10, y), font, font_scale, font_color, thickness, line_type)
                # 
                # combined_frame = np.hstack((frame, white_box, plot_image_rgb))
                combined_frame = np.hstack((frame, white_box, plot_image_rgb))
            else:
                # combined_frame = np.hstack((frame, plot_image_rgb))
                combined_frame = np.hstack((frame, plot_image_rgb))
            # 
            combined_frame = combined_frame.astype(np.uint8)
            # 
            if task_description is not None:
                # ---- create white top bar ----
                top_bar = np.ones((top_pad, combined_frame.shape[1], 3), dtype=np.uint8) * 255
                # ---- add text ----
                label = f'Task: "{task_description}"'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                padding = 12
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = padding
                text_y = padding + text_h
                cv2.putText(
                    top_bar,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),  # black text
                    thickness,
                    cv2.LINE_AA,
                )
                # ---- stack on top of frame ----
                combined_frame = np.vstack((top_bar, combined_frame))
            # 
            assert combined_frame.shape[0] == output_height
            assert combined_frame.shape[1] == output_width
            assert combined_frame.dtype == np.uint8
            # out.write(combined_frame)
            out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            frame_number += 1
            data_point_number += 1
        # 
        out.release()
    plt.close(fig)   
    del canvas 
    # os.system(f"ffmpeg -y -i {plot_save_path} -c:v libx264 -pix_fmt yuv420p {plot_save_path.replace('.avi', '.mp4')}")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    output_path = plot_save_path.replace(".avi", ".mp4")
    os.system(
        f'"{ffmpeg_exe}" -y '
        f'-i "{plot_save_path}" '
        f'-r {output_fps:.12g} '
        f'-c:v libx264 '
        f'-pix_fmt yuv420p '
        f'"{output_path}"'
    )
    os.remove(plot_save_path)
    if return_frames:
        return frame_list




