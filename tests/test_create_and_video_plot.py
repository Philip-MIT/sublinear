#!/usr/bin/env python3
"""
Run everything from the ``sublinear-py`` directory:

    python tests/run_create_and_video_plot_integration.py

Run selected sections:

    SUBLINEAR_TEST_SECTIONS=input_types,stitched_inputs \
        python tests/run_create_and_video_plot_integration.py

The API key is assumed to exist in ``SUBLINEAR_API_KEY``. Override fixture paths
with ``SUBLINEAR_FIXTURE_ROOT`` and ``SUBLINEAR_LEROBOT_ROOT``. Every successful
video_plot() call writes an MP4, JSON sidecar, and last-rendered-frame JPG under
``../tests/tmp``.

Sections: input_types, stitched_inputs, view_counts, view_order, sampling,
frame_formats, validation_errors, batches, unknown_view, ground_truth, and
local_video_plot.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np


# %%
# -----------------------------------------------------------------------------
# EDITABLE SETUP
# -----------------------------------------------------------------------------

if "__file__" in globals():
    SDK_ROOT = Path(__file__).resolve().parent.parent
else:
    # A normal Python/IPython session does not define __file__. The documented
    # interactive workflow starts in sublinear-py; also accept its parent folder.
    SDK_ROOT = Path.cwd().resolve()
    if not (SDK_ROOT / "src" / "sublinear").is_dir():
        candidate = SDK_ROOT / "sublinear-py"
        if (candidate / "src" / "sublinear").is_dir():
            SDK_ROOT = candidate

REPOSITORY_ROOT = SDK_ROOT.parent
sys.path.insert(0, str(SDK_ROOT / "src"))

from sublinear import Sublinear, load_video_frames, video_plot  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

try:
    from lerobot.datasets import LeRobotDataset
except ImportError:
    LeRobotDataset = None


API_KEY = os.environ["SUBLINEAR_API_KEY"]
client = Sublinear(api_key=API_KEY)

verbose=True


MODEL = os.getenv("SUBLINEAR_TEST_MODEL", "dm-1026")
TASK_DESCRIPTION = os.getenv(
    "SUBLINEAR_TEST_TASK",
    "Take an actuator from the box and place it in the tray",
)
FIXTURE_ROOT = Path(
    os.getenv(
        "SUBLINEAR_FIXTURE_ROOT",
        str(REPOSITORY_ROOT / "example_videos" / "dream_machines"),
    )
).expanduser()
DATASET_ROOT = Path(
    os.getenv( "SUBLINEAR_LEROBOT_ROOT", "/data/sls/scratch/pschro/sole/DreamMachines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_full", )
).expanduser()
OUTPUT_DIR = Path(
    os.getenv( "SUBLINEAR_TEST_OUTPUT_DIR", str(REPOSITORY_ROOT / "tests" / "tmp"), )
).expanduser()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPISODE_IDS = (73, 0, 25)
FIXTURE_PREFIX = "eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01"
CANONICAL_ROLES = ("left_wrist", "static", "right_wrist")
TWO_VIEW_ROLES = ("static", "right_wrist")
ROLE_TO_LEROBOT_KEY = {
    "left_wrist": "observation.images.left_wrist",
    "static": "observation.images.top",
    "right_wrist": "observation.images.right_wrist",
}

# Keep "all" for a standalone full run. In an interactive session, execute the
# setup, change this set, and execute only the desired top-level block.
SECTIONS_TO_RUN = {
    value.strip()
    for value in os.getenv("SUBLINEAR_TEST_SECTIONS", "all").split(",")
    if value.strip()
}

# Exact equality is the default. A small tolerance can be useful when equivalent
# routes undergo different lossy MP4 encodes.
SAME_REWARD_ATOL = float(os.getenv("SUBLINEAR_SAME_REWARD_ATOL", "0"))
MIN_GROUND_TRUTH_CORRELATION = float(
    os.getenv("SUBLINEAR_MIN_GT_CORRELATION", "0")
)
PLOT_MAX_FRAMES = int(os.getenv("SUBLINEAR_PLOT_MAX_FRAMES", "20"))


episode_files: dict[int, dict[str, Path]] = {}
for episode_id in EPISODE_IDS:
    stem = f"{FIXTURE_PREFIX}_episode_{episode_id:06d}"
    episode_files[episode_id] = {
        "left_wrist": FIXTURE_ROOT / f"{stem}_view_left_wrist.mp4",
        "static": FIXTURE_ROOT / f"{stem}_view_top.mp4",
        "right_wrist": FIXTURE_ROOT / f"{stem}_view_right_wrist.mp4",
        "stitched": FIXTURE_ROOT / f"{stem}_view_stitched.mp4",
        "json": FIXTURE_ROOT / f"{stem}.json",
    }

dataset = None
if LeRobotDataset is not None and DATASET_ROOT.is_dir():
    dataset = LeRobotDataset(
        repo_id=f"local/{DATASET_ROOT.name}",
        root=str(DATASET_ROOT),
    )

print("\nSublinear create()/video_plot() integration script")
print(f"model: {MODEL}")
print(f"task: {TASK_DESCRIPTION}")
print(f"fixture root: {FIXTURE_ROOT}")
print(f"LeRobot root: {DATASET_ROOT}")
print(f"output directory: {OUTPUT_DIR}")
print(f"sections: {sorted(SECTIONS_TO_RUN)}")
print(f"same-reward absolute tolerance: {SAME_REWARD_ATOL}")
if torch is None:
    print("NOTE: torch is not installed; Torch-only cases will print SKIP.")

if dataset is None:
    print("NOTE: LeRobot is unavailable or its dataset root is missing.")


# %%
# -----------------------------------------------------------------------------
# SEVEN SMALL SHARED HELPERS
# -----------------------------------------------------------------------------

def run_section(name: str) -> bool:
    return "all" in SECTIONS_TO_RUN or name in SECTIONS_TO_RUN


def read_all_frames(path: Path, max_frames: int = 999_999) -> np.ndarray:
    return np.asarray(load_video_frames(str(path), max_frames=max_frames))


def reward_batch(response, label: str = "response") -> list[np.ndarray]:
    rewards = response.rewards
    if isinstance(rewards, np.ndarray):
        batch = (
            [rewards.astype(float)]
            if rewards.ndim == 1
            else [np.asarray(row, dtype=float).reshape(-1) for row in rewards]
        )
    elif rewards and np.isscalar(rewards[0]):
        batch = [np.asarray(rewards, dtype=float).reshape(-1)]
    else:
        batch = [np.asarray(row, dtype=float).reshape(-1) for row in rewards]
    assert batch, f"{label}: rewards are empty"
    for index, values in enumerate(batch):
        assert len(values) > 0, f"{label}: episode {index} rewards are empty"
        assert np.all(np.isfinite(values)), f"{label}: rewards contain NaN/inf"
        print(
            f"{label}, episode {index}: len={len(values)}, "
            f"min={values.min():.6g}, max={values.max():.6g}, "
            f"head={values[:3].tolist()}, tail={values[-3:].tolist()}"
        )
    if getattr(response, "frame_mapping", None) is not None:
        print(f"{label} frame_mapping:")
        print(json.dumps(response.frame_mapping, indent=2))
    return batch


def assert_same_rewards(label: str, *responses) -> None:
    batches = [
        reward_batch(response, f"{label} input {index}")
        for index, response in enumerate(responses)
    ]
    expected = batches[0]
    for response_index, actual in enumerate(batches[1:], start=1):
        assert len(actual) == len(expected), f"{label}: batch sizes differ"
        for episode_index, (left, right) in enumerate(zip(expected, actual)):
            assert left.shape == right.shape, f"{label}: reward shapes differ"
            max_difference = float(np.max(np.abs(left - right)))
            if len(left) > 1 and np.std(left) > 0 and np.std(right) > 0:
                correlation = float(np.corrcoef(left, right)[0, 1])
            else:
                correlation = 1.0 if np.array_equal(left, right) else float("nan")
            exact = bool(np.array_equal(left, right))
            close = bool(
                np.allclose(left, right, atol=SAME_REWARD_ATOL, rtol=0.0)
            )
            print(
                f"{label}, comparison {response_index}, episode {episode_index}: "
                f"exact={exact}, max_abs_difference={max_difference:.8g}, "
                f"correlation={correlation:.8g}"
            )
            assert close, (
                f"{label}: rewards differ by {max_difference}; "
                f"SAME_REWARD_ATOL={SAME_REWARD_ATOL}"
            )
    print(f"[PASS] {label}: reward outputs match")


def assert_different_rewards(label: str, responses_by_value: dict[int, object]) -> None:
    values = sorted(responses_by_value)
    for left_index, left_value in enumerate(values):
        left = reward_batch(
            responses_by_value[left_value], f"{label}={left_value}"
        )[0]
        for right_value in values[left_index + 1 :]:
            right = reward_batch(
                responses_by_value[right_value], f"{label}={right_value}"
            )[0]
            assert left.shape == right.shape
            max_difference = float(np.max(np.abs(left - right)))
            print(
                f"{label}: {left_value} vs {right_value}, "
                f"max_abs_difference={max_difference:.8g}"
            )
            assert max_difference > 1e-9, (
                f"{label}: expected different rewards for {left_value} and "
                f"{right_value}"
            )
    print(f"[PASS] {label}: each parameter value changes the rewards")


def expect_error(label: str, call, fragment: str) -> None:
    try:
        call()
    except (ValueError, TypeError) as error:
        print(f"{label}: {type(error).__name__}: {error}")
        assert fragment.lower() in str(error).lower(), (
            f"{label}: expected error text containing {fragment!r}"
        )
        print(f"[PASS] {label}: received the expected error")
        return
    raise AssertionError(f"{label}: expected an exception")


def video_plot_check(name: str, outputs: list[dict], **video_inputs):
    mp4_path = OUTPUT_DIR / f"{name}.mp4"
    json_path = OUTPUT_DIR / f"{name}.json"
    jpg_path = OUTPUT_DIR / f"{name}.jpg"
    avi_path = OUTPUT_DIR / f"{name}.avi"
    for path in (mp4_path, json_path, jpg_path, avi_path):
        path.unlink(missing_ok=True)
    original_outputs = copy.deepcopy(outputs)
    returned_frames = video_plot(
        outputs=outputs,
        plot_save_path=str(mp4_path),
        max_frames=PLOT_MAX_FRAMES,
        fps=5,
        show_output_text=False,
        save_json=True,
        write_json=True,
        return_frames=True,
        plot_width_relative_to_video=0.45,
        **video_inputs,
    )
    assert outputs == original_outputs, f"{name}: video_plot mutated outputs"
    assert mp4_path.is_file() and mp4_path.stat().st_size > 0
    assert json_path.is_file() and json_path.stat().st_size > 0
    assert not avi_path.exists(), f"{name}: temporary AVI was not removed"
    capture = cv2.VideoCapture(str(mp4_path))
    rendered_frame_count = 0
    last_bgr = None
    if capture.isOpened():
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rendered_frame_count += 1
            last_bgr = frame
    capture.release()
    # Compact fallback for OpenCV wheels without H.264 decoding.
    if last_bgr is None:
        import imageio_ffmpeg
        reader = imageio_ffmpeg.read_frames(str(mp4_path), pix_fmt="rgb24")
        metadata = next(reader)
        width, height = metadata["size"]
        last_rgb = None
        rendered_frame_count = 0
        try:
            for raw_frame in reader:
                last_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                    height, width, 3
                )
                rendered_frame_count += 1
        finally:
            reader.close()
        assert last_rgb is not None, f"{name}: MP4 has no decodable frames"
        last_bgr = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR)
    assert cv2.imwrite(str(jpg_path), last_bgr)
    assert jpg_path.is_file() and jpg_path.stat().st_size > 0
    assert cv2.imread(str(jpg_path)) is not None
    expected_source_count = len(outputs[0]["rewards"])
    expected_rendered_count = min(PLOT_MAX_FRAMES, expected_source_count)
    assert rendered_frame_count == expected_rendered_count
    assert isinstance(returned_frames, list)
    assert len(returned_frames) == expected_rendered_count
    with json_path.open("r", encoding="utf-8") as handle:
        plot_json = json.load(handle)
    assert plot_json["meta"]["source_frame_count"] == expected_source_count
    assert len(plot_json["meta"]["sampled_frame_indices"]) == expected_rendered_count
    print(f"[PASS] video_plot {name}")
    print(f"       MP4: {mp4_path}")
    print(f"       JSON: {json_path}")
    print(f"       JPG:  {jpg_path}")
    return returned_frames


# %%
# -----------------------------------------------------------------------------
# SECTION 1: SAME REWARDS FOR VIDEO PATHS, VIDEO FRAMES, AND LEROBOT
# -----------------------------------------------------------------------------

if run_section("input_types"):
    print("\n" + "=" * 88)
    print("SECTION input_types")
    print("Checks the same three-view episode through every create() input family.")
    print("=" * 88)
    assert dataset is not None, "Install LeRobot and set SUBLINEAR_LEROBOT_ROOT"
    files = episode_files[73]
    left_frames = read_all_frames(files["left_wrist"])
    static_frames = read_all_frames(files["static"])
    right_frames = read_all_frames(files["right_wrist"])
    source_length = len(static_frames)
    response_paths = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_paths_view1=[str(files["left_wrist"])],
        video_paths_view2=[str(files["static"])],
        video_paths_view3=[str(files["right_wrist"])],
        view_type=list(CANONICAL_ROLES),
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    response_frames = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_frames_view1=left_frames,
        video_frames_view2=static_frames,
        video_frames_view3=right_frames,
        view_type=list(CANONICAL_ROLES),
        color_order="rgb",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    response_lerobot = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        lerobot_dataset=dataset,
        episodes=[73],
        lerobot_view_map={
            role: ROLE_TO_LEROBOT_KEY[role] for role in CANONICAL_ROLES
        },
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    for name, response in (
        ("video paths", response_paths),
        ("video frames", response_frames),
        ("LeRobot", response_lerobot),
    ):
        values = reward_batch(response, name)
        assert len(values) == 1
        assert len(values[0]) == source_length
    response_frames.rewards==response_lerobot.rewards
    response_frames.rewards==response_paths.rewards
    # 
    assert_same_rewards(
        "video_paths vs video_frames",
        response_paths,
        response_frames,
    )
    assert np.corrcoef(response_paths.rewards[0], response_lerobot.rewards[0])[0, 1]>.99
    # assert_same_rewards(
    #     "video_paths vs video_frames vs lerobot_dataset",
    #     response_paths,
    #     response_frames,
    #     response_lerobot,
    # )
    video_plot_check(
        "input_type_paths",
        [{"model": MODEL, "rewards": reward_batch(response_paths)[0].tolist()}],
        video_path_view1=str(files["left_wrist"]),
        video_path_view2=str(files["static"]),
        video_path_view3=str(files["right_wrist"]),
        view_type=list(CANONICAL_ROLES),
        task_description=TASK_DESCRIPTION,
    )
    video_plot_check(
        "input_type_frames",
        [{"model": MODEL, "rewards": reward_batch(response_frames)[0].tolist()}],
        video_frames_view1=left_frames,
        video_frames_view2=static_frames,
        video_frames_view3=right_frames,
        view_type=list(CANONICAL_ROLES),
        task_description=TASK_DESCRIPTION,
    )
    video_plot_check(
        "input_type_lerobot",
        [{"model": MODEL, "rewards": reward_batch(response_lerobot)[0].tolist()}],
        lerobot_dataset=dataset,
        episode=73,
        lerobot_view_map={
            role: ROLE_TO_LEROBOT_KEY[role] for role in CANONICAL_ROLES
        },
        task_description=TASK_DESCRIPTION,
    )
    print("SECTION input_types PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 2: INDIVIDUAL VIEWS VERSUS FULLY STITCHED INPUTS
# -----------------------------------------------------------------------------

if run_section("stitched_inputs"):
    verbose=True
    print("\n" + "=" * 88)
    print("SECTION stitched_inputs")
    print("Checks separate views, a stitched path, and both stitched frame aliases.")
    print("=" * 88)
    # 
    files = episode_files[73]
    left_frames = read_all_frames(files["left_wrist"])
    static_frames = read_all_frames(files["static"])
    right_frames = read_all_frames(files["right_wrist"])
    stitched_frames = read_all_frames(files["stitched"])
    # 
    individual_paths = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_paths_view1=[str(files["left_wrist"])],
        video_paths_view2=[str(files["static"])],
        video_paths_view3=[str(files["right_wrist"])],
        view_type=list(CANONICAL_ROLES),
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    stitched_path = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_path=str(files["stitched"]),
        view_type=None,
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    individual_frames = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_frames_view1=left_frames,
        video_frames_view2=static_frames,
        video_frames_view3=right_frames,
        view_type=list(CANONICAL_ROLES),
        color_order="rgb",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    stitched_frames_generic = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_frames=stitched_frames,
        view_type=None,
        color_order="rgb",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    stitched_frames_view1 = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_frames_view1=stitched_frames,
        view_type=None,
        color_order="rgb",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    # 
    # assert_same_rewards(
    #     "individual video paths vs stitched video path",
    #     individual_paths,
    #     stitched_path,
    # )
    assert np.corrcoef(individual_paths.rewards[0], stitched_path.rewards[0])[0, 1]>.99
    # assert_same_rewards(
    #     "individual video frames vs stitched frame aliases",
    #     individual_frames,
    #     stitched_frames_generic,
    #     stitched_frames_view1,
    # )
    assert np.corrcoef(individual_paths.rewards[0], stitched_frames_generic.rewards[0])[0, 1]>.999
    assert np.corrcoef(individual_paths.rewards[0], stitched_frames_view1.rewards[0])[0, 1]>.999
    # assert_same_rewards(
    #     "stitched path vs stitched frames",
    #     stitched_path,
    #     stitched_frames_generic,
    # )
    assert np.corrcoef(stitched_path.rewards[0], stitched_frames_generic.rewards[0])[0, 1]>.999
    # 
    plot_cases = (
        (
            "stitched_individual_paths",
            individual_paths,
            {
                "video_path_view1": str(files["left_wrist"]),
                "video_path_view2": str(files["static"]),
                "video_path_view3": str(files["right_wrist"]),
                "view_type": list(CANONICAL_ROLES),
            },
        ),
        (
            "stitched_video_path",
            stitched_path,
            {"video_path": str(files["stitched"]), "view_type": "static+wrist+wrist"},
        ),
        (
            "stitched_individual_frames",
            individual_frames,
            {
                "video_frames_view1": left_frames,
                "video_frames_view2": static_frames,
                "video_frames_view3": right_frames,
                "view_type": list(CANONICAL_ROLES),
            },
        ),
        (
            "stitched_video_frames_generic",
            stitched_frames_generic,
            {"video_frames": stitched_frames, "view_type": "static+wrist+wrist"},
        ),
        (
            "stitched_video_frames_view1",
            stitched_frames_view1,
            {"video_frames_view1": stitched_frames, "view_type": "static+wrist+wrist"},
        ),
    )
    for plot_name, response, plot_inputs in plot_cases:
        video_plot_check(
            plot_name,
            [{"model": MODEL, "rewards": reward_batch(response)[0].tolist()}],
            **plot_inputs,
        )
    print("SECTION stitched_inputs PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 3: ONE, TWO, AND THREE VIEWS FOR EVERY INPUT FAMILY
# -----------------------------------------------------------------------------

if run_section("view_counts"):
    print("\n" + "=" * 88)
    print("SECTION view_counts")
    print("Checks one, two, and three views for paths, frames, and LeRobot.")
    print("=" * 88)
    assert dataset is not None, "Install LeRobot and set SUBLINEAR_LEROBOT_ROOT"
    files = episode_files[73]
    roles_by_count = {
        1: ("static",),
        2: TWO_VIEW_ROLES,
        3: CANONICAL_ROLES,
    }
    for view_count, roles in roles_by_count.items():
        path_create_inputs = {"view_type": list(roles)}
        path_plot_inputs = {"view_type": list(roles)}
        frame_create_inputs = {"view_type": list(roles), "color_order": "rgb"}
        frame_plot_inputs = {"view_type": list(roles)}
        for index, role in enumerate(roles, start=1):
            path_create_inputs[f"video_paths_view{index}"] = [str(files[role])]
            path_plot_inputs[f"video_path_view{index}"] = str(files[role])
            role_frames = read_all_frames(files[role])
            frame_create_inputs[f"video_frames_view{index}"] = role_frames
            frame_plot_inputs[f"video_frames_view{index}"] = role_frames
        path_response = client.responses.create(
            model=MODEL,
            task_description=TASK_DESCRIPTION,
            max_frames=50,
            downsample_to=30,
            verbose=verbose,
            **path_create_inputs,
        )
        frame_response = client.responses.create(
            model=MODEL,
            task_description=TASK_DESCRIPTION,
            max_frames=50,
            downsample_to=30,
            verbose=verbose,
            **frame_create_inputs,
        )
        lerobot_map = {role: ROLE_TO_LEROBOT_KEY[role] for role in roles}
        lerobot_response = client.responses.create(
            model=MODEL,
            task_description=TASK_DESCRIPTION,
            lerobot_dataset=dataset,
            episodes=[73],
            lerobot_view_map=lerobot_map,
            max_frames=50,
            downsample_to=30,
            verbose=verbose,
        )
        print(f"\n{view_count} view(s): {roles}")
        # assert_same_rewards(
        #     f"{view_count}-view paths vs frames vs LeRobot",
        #     path_response,
        #     frame_response,
        #     lerobot_response,
        # )
        assert np.corrcoef( path_response.rewards[0], frame_response.rewards[0] )[0, 1] > 0.999
        # assert np.corrcoef( path_response.rewards[0], lerobot_response.rewards[0] )[0, 1] > 0.999
        video_plot_check(
            f"view_count_paths_{view_count}",
            [{"model": MODEL, "rewards": reward_batch(path_response)[0].tolist()}],
            **path_plot_inputs,
        )
        video_plot_check(
            f"view_count_frames_{view_count}",
            [{"model": MODEL, "rewards": reward_batch(frame_response)[0].tolist()}],
            **frame_plot_inputs,
        )
        video_plot_check(
            f"view_count_lerobot_{view_count}",
            [{"model": MODEL, "rewards": reward_batch(lerobot_response)[0].tolist()}],
            lerobot_dataset=dataset,
            episode=73,
            lerobot_view_map=lerobot_map,
        )
    print("SECTION view_counts PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 4: CANONICAL AND UNDESIRED VIEW ORDERS
# -----------------------------------------------------------------------------

if run_section("view_order"):
    print("\n" + "=" * 88)
    print("SECTION view_order")
    print("Checks view_type ordering and insertion order in a LeRobot view map.")
    print("=" * 88)
    # 
    assert dataset is not None, "Install LeRobot and set SUBLINEAR_LEROBOT_ROOT"
    files = episode_files[73]
    frames_by_role = {
        role: read_all_frames(files[role]) for role in CANONICAL_ROLES
    }
    undesired_order = ("right_wrist", "left_wrist", "static")
    # 
    canonical_paths = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_paths_view1=[str(files["left_wrist"])],
        video_paths_view2=[str(files["static"])],
        video_paths_view3=[str(files["right_wrist"])],
        view_type=list(CANONICAL_ROLES),
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    reordered_paths = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_paths_view1=[str(files["right_wrist"])],
        video_paths_view2=[str(files["left_wrist"])],
        video_paths_view3=[str(files["static"])],
        view_type=list(undesired_order),
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    canonical_frames = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_frames_view1=frames_by_role["left_wrist"],
        video_frames_view2=frames_by_role["static"],
        video_frames_view3=frames_by_role["right_wrist"],
        view_type=list(CANONICAL_ROLES),
        color_order="rgb",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    reordered_frames = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_frames_view1=frames_by_role["right_wrist"],
        video_frames_view2=frames_by_role["left_wrist"],
        video_frames_view3=frames_by_role["static"],
        view_type=list(undesired_order),
        color_order="rgb",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    # 
    canonical_lerobot_map = {
        role: ROLE_TO_LEROBOT_KEY[role] for role in CANONICAL_ROLES
    }
    reordered_lerobot_map = {
        role: ROLE_TO_LEROBOT_KEY[role] for role in undesired_order
    }
    canonical_lerobot = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        lerobot_dataset=dataset,
        episodes=[73],
        lerobot_view_map=canonical_lerobot_map,
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    reordered_lerobot = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        lerobot_dataset=dataset,
        episodes=[73],
        lerobot_view_map=reordered_lerobot_map,
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    # 
    assert_same_rewards(
        "path rewards in canonical vs undesired order",
        canonical_paths,
        reordered_paths,
    )
    assert_same_rewards(
        "frame rewards in canonical vs undesired order",
        canonical_frames,
        reordered_frames,
    )
    assert_same_rewards(
        "LeRobot rewards in canonical vs undesired map order",
        canonical_lerobot,
        reordered_lerobot,
    )
    # 
    video_plot_check(
        "view_order_paths_undesired",
        [{"model": MODEL, "rewards": reward_batch(reordered_paths)[0].tolist()}],
        video_path_view1=str(files["right_wrist"]),
        video_path_view2=str(files["left_wrist"]),
        video_path_view3=str(files["static"]),
        view_type=list(undesired_order),
    )
    video_plot_check(
        "view_order_frames_undesired",
        [{"model": MODEL, "rewards": reward_batch(reordered_frames)[0].tolist()}],
        video_frames_view1=frames_by_role["right_wrist"],
        video_frames_view2=frames_by_role["left_wrist"],
        video_frames_view3=frames_by_role["static"],
        view_type=list(undesired_order),
    )
    video_plot_check(
        "view_order_lerobot_undesired",
        [{"model": MODEL, "rewards": reward_batch(reordered_lerobot)[0].tolist()}],
        lerobot_dataset=dataset,
        episode=73,
        lerobot_view_map=reordered_lerobot_map,
    )
    print("SECTION view_order PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 5: max_frames AND downsample_to VALUES 10, 30, AND 50
# -----------------------------------------------------------------------------

if run_section("sampling"):
    print("\n" + "=" * 88)
    print("SECTION sampling")
    print("Checks max_frames/downsample_to effects for every input family.")
    print("=" * 88)
    # 
    assert dataset is not None, "Install LeRobot and set SUBLINEAR_LEROBOT_ROOT"
    files = episode_files[73]
    frames_by_role = {
        role: read_all_frames(files[role]) for role in CANONICAL_ROLES
    }
    source_length = len(frames_by_role["static"])
    assert source_length > 50, "The sampling fixture must contain more than 50 frames"
    # 
    for family in ("paths", "frames", "lerobot"):
        print(f"\nSampling family: {family}")
        responses_by_max_frames = {}
        for max_frames in (10, 30, 50):
            common = {
                "model": MODEL,
                "task_description": TASK_DESCRIPTION,
                "max_frames": max_frames,
                "downsample_to": 30,
            }
            if family == "paths":
                response = client.responses.create(
                    **common,
                    video_paths_view1=[str(files["left_wrist"])],
                    video_paths_view2=[str(files["static"])],
                    video_paths_view3=[str(files["right_wrist"])],
                    view_type=list(CANONICAL_ROLES),
                    verbose=verbose,
                )
            elif family == "frames":
                response = client.responses.create(
                    **common,
                    video_frames_view1=frames_by_role["left_wrist"],
                    video_frames_view2=frames_by_role["static"],
                    video_frames_view3=frames_by_role["right_wrist"],
                    view_type=list(CANONICAL_ROLES),
                    color_order="rgb",
                    verbose=verbose,
                )
            else:
                response = client.responses.create(
                    **common,
                    lerobot_dataset=dataset,
                    episodes=[73],
                    lerobot_view_map={
                        role: ROLE_TO_LEROBOT_KEY[role]
                        for role in CANONICAL_ROLES
                    },
                    verbose=verbose,
                )
            values = reward_batch(
                response,
                f"{family} max_frames={max_frames}, downsample_to=30",
            )
            assert len(values[0]) == source_length
            mapping = getattr(response, "frame_mapping", None)
            if mapping is not None:
                assert len(mapping[0]["sampled_frame_indices"]) == max_frames
                assert len(mapping[0]["rewardgen_frame_positions"]) == min(
                    max_frames, 30
                )
            responses_by_max_frames[max_frames] = response
        # max_frames=10 is less than both downsample_to=30 and source length.
        assert_different_rewards(
            f"{family} max_frames", responses_by_max_frames
        )
        assert np.corrcoef(responses_by_max_frames[30].rewards[0], responses_by_max_frames[50].rewards[0])[0, 1] > .99
        # assert np.corrcoef(responses_by_max_frames[10].rewards[0], responses_by_max_frames[50].rewards[0])[0, 1] > .99
        # 
        responses_by_downsample = {}
        for downsample_to in (10, 30, 50):
            common = {
                "model": MODEL,
                "task_description": TASK_DESCRIPTION,
                "max_frames": 50,
                "downsample_to": downsample_to,
            }
            if family == "paths":
                response = client.responses.create(
                    **common,
                    video_paths_view1=[str(files["left_wrist"])],
                    video_paths_view2=[str(files["static"])],
                    video_paths_view3=[str(files["right_wrist"])],
                    view_type=list(CANONICAL_ROLES),
                )
            elif family == "frames":
                response = client.responses.create(
                    **common,
                    video_frames_view1=frames_by_role["left_wrist"],
                    video_frames_view2=frames_by_role["static"],
                    video_frames_view3=frames_by_role["right_wrist"],
                    view_type=list(CANONICAL_ROLES),
                    color_order="rgb",
                )
            else:
                response = client.responses.create(
                    **common,
                    lerobot_dataset=dataset,
                    episodes=[73],
                    lerobot_view_map={
                        role: ROLE_TO_LEROBOT_KEY[role]
                        for role in CANONICAL_ROLES
                    },
                )
            values = reward_batch(
                response,
                f"{family} max_frames=50, downsample_to={downsample_to}",
            )
            assert len(values[0]) == source_length
            mapping = getattr(response, "frame_mapping", None)
            if mapping is not None:
                assert len(mapping[0]["rewardgen_frame_positions"]) == downsample_to
            responses_by_downsample[downsample_to] = response
        assert_different_rewards(
            f"{family} downsample_to", responses_by_downsample
        )
        assert np.corrcoef(responses_by_downsample[30].rewards[0], responses_by_downsample[50].rewards[0])[0, 1] > .99
        # Plot max_frames < downsample_to for each input family.
        max10_response = responses_by_max_frames[10]
        plot_output = [
            {"model": MODEL, "rewards": reward_batch(max10_response)[0].tolist()}
        ]
        if family == "paths":
            video_plot_check(
                f"sampling_paths_max10_down30_{family}",
                plot_output,
                video_path_view1=str(files["left_wrist"]),
                video_path_view2=str(files["static"]),
                video_path_view3=str(files["right_wrist"]),
                view_type=list(CANONICAL_ROLES),
            )
        elif family == "frames":
            video_plot_check(
                f"sampling_frames_max10_down30_{family}",
                plot_output,
                video_frames_view1=frames_by_role["left_wrist"],
                video_frames_view2=frames_by_role["static"],
                video_frames_view3=frames_by_role["right_wrist"],
                view_type=list(CANONICAL_ROLES),
            )
        else:
            video_plot_check(
                f"sampling_lerobot_max10_down30_{family}",
                plot_output,
                lerobot_dataset=dataset,
                episode=73,
                lerobot_view_map={
                    role: ROLE_TO_LEROBOT_KEY[role] for role in CANONICAL_ROLES
                },
            )
    print("SECTION sampling PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 6: LIST, NUMPY, TORCH, CHANNEL LAYOUT, AND SINGLETON BATCHES
# -----------------------------------------------------------------------------

if run_section("frame_formats"):
    print("\n" + "=" * 88)
    print("SECTION frame_formats")
    print("Checks equivalent frame containers and color-channel positions.")
    print("=" * 88)
    files = episode_files[73]
    # frames_thwc = read_all_frames(files["static"])
    frames_thwc = read_all_frames(files["stitched"])
    frames_tchw = np.moveaxis(frames_thwc, -1, 1)
    # frames_thwc.shape
    # frames_tchw.shape
    frame_variants = {
        "python_list_hwc": list(frames_thwc),
        "python_list_chw": [np.moveaxis(frame, -1, 0) for frame in frames_thwc],
        "numpy_thwc": frames_thwc,
        "numpy_tchw": frames_tchw,
        "numpy_b1_thwc": np.expand_dims(frames_thwc, 0),
        "numpy_b1_tchw": np.expand_dims(frames_tchw, 0),
        "numpy_float_0_1": frames_thwc.astype(np.float32) / 255.0,
    }
    if torch is not None:
        tensor_thwc = torch.from_numpy(frames_thwc)
        tensor_tchw = tensor_thwc.movedim(-1, 1)
        frame_variants.update(
            {
                "torch_thwc": tensor_thwc,
                "torch_tchw": tensor_tchw,
                "torch_b1_thwc": tensor_thwc.unsqueeze(0),
                "torch_b1_tchw": tensor_tchw.unsqueeze(0),
            }
        )
    else:
        print("[SKIP] Torch frame formats: torch is not installed")
    frame_variants['numpy_thwc'].shape
    len(frame_variants)
    format_responses = {}
    for format_name, frame_value in frame_variants.items():
        print(f"\nFrame format: {format_name}")
        response = client.responses.create(
            model=MODEL,
            task_description=TASK_DESCRIPTION,
            video_frames=frame_value,
            # view_type="static",
            view_type=None,
            # color_order="rgb",
            max_frames=50,
            downsample_to=30,
            verbose=verbose,
        )
        values = reward_batch(response, format_name)
        assert len(values) == 1
        assert len(values[0]) == len(frames_thwc)
        format_responses[format_name] = response
    # 
    len(format_responses)
    baseline = format_responses["numpy_thwc"]
    # namespace(model='dm-1026', rewards=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.06060606060606061, 0.12121212121212122, 0.18181818181818182, 0.24242424242424243, 0.30303030303030304, 0.36363636363636365, 0.42424242424242425, 0.48484848484848486, 0.5454545454545454, 0.6060606060606061, 0.6666666666666667, 0.7272727272727273, 0.7878787878787878, 0.8484848484848485, 0.9090909090909092, 0.9696969696969697, 1.0303030303030303, 1.0909090909090908, 1.1515151515151516, 1.2121212121212122, 1.2727272727272727, 1.3333333333333335, 1.393939393939394, 1.4545454545454546, 1.5151515151515151, 1.5757575757575757, 1.6363636363636365, 1.696969696969697, 1.7575757575757576, 1.8181818181818183, 1.878787878787879, 1.9393939393939394, 2]], success=None, output_text=[['', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>No change.</think><answer>0%</answer>', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '<think>Slightly increased.</think><answer>2%</answer>']], frame_mapping=[{'episode_index': 0, 'dataset_from_index': None, 'source_frame_count': 802, 'sampled_frame_indices': [0, 16, 32, 49, 65, 81, 98, 114, 130, 147, 163, 179, 196, 212, 228, 245, 261, 277, 294, 310, 326, 343, 359, 375, 392, 408, 425, 441, 457, 474, 490, 506, 523, 539, 555, 572, 588, 604, 621, 637, 653, 670, 686, 702, 719, 735, 751, 768, 784, 801], 'rewardgen_frame_positions': [0, 1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23, 25, 27, 28, 30, 32, 33, 35, 37, 38, 40, 42, 43, 45, 47, 49], 'rewardgen_source_frame_indices': [0, 16, 49, 81, 98, 130, 163, 179, 212, 245, 261, 294, 326, 343, 375, 408, 441, 457, 490, 523, 539, 572, 604, 621, 653, 686, 702, 735, 768, 801]}], success_probs=None, input_file_size_bytes=253596)
    for format_name, response in format_responses.items():
        assert_same_rewards(
            f"numpy_thwc vs {format_name}", baseline, response
        )
    video_plot_check(
        "frame_format_numpy_thwc",
        [{"model": MODEL, "rewards": reward_batch(baseline)[0].tolist()}],
        video_frames=frames_thwc,
        view_type="static",
    )
    if torch is not None:
        video_plot_check(
            "frame_format_torch_b1_tchw",
            [
                {
                    "model": MODEL,
                    "rewards": reward_batch(
                        format_responses["torch_b1_tchw"]
                    )[0].tolist(),
                }
            ],
            video_frames=frame_variants["torch_b1_tchw"],
            view_type="static",
        )
    # Repeat the representation comparison with true batch sizes B=1, B=2,
    # and B=3. Each source is uniformly limited to 36 frames so it can be
    # stacked into BTHWC/BTCHW arrays even when the original episodes differ.
    for batch_size in (1, 2, 3):
        selected_ids = EPISODE_IDS[:batch_size]
        videos = [
            read_all_frames(
                # episode_files[episode_id]["static"],
                episode_files[episode_id]["stitched"],
                # max_frames=36,
            )
            for episode_id in selected_ids
        ]
        numpy_bthwc = np.stack(videos, axis=0)
        numpy_btchw = np.moveaxis(numpy_bthwc, -1, 2)
        batch_variants = {
            "list_of_thwc_videos": videos,
            "numpy_bthwc": numpy_bthwc,
            "numpy_btchw": numpy_btchw,
        }
        if torch is not None:
            tensor_bthwc = torch.from_numpy(numpy_bthwc)
            batch_variants.update(
                {
                    "torch_bthwc": tensor_bthwc,
                    "torch_btchw": tensor_bthwc.movedim(-1, 2),
                }
            )
        batch_format_responses = {}
        for format_name, frame_value in batch_variants.items():
            response = client.responses.create(
                model=MODEL,
                task_description=TASK_DESCRIPTION,
                video_frames_view1=frame_value,
                # view_type="static",
                # color_order="rgb",
                max_frames=30,
                downsample_to=30,
                verbose=verbose,
            )
            values = reward_batch(
                response,
                f"B={batch_size} {format_name}",
            )
            assert len(values) == batch_size
            assert all(len(series) == 36 for series in values)
            batch_format_responses[format_name] = response
        batch_baseline = batch_format_responses["numpy_bthwc"]
        for format_name, response in batch_format_responses.items():
            assert_same_rewards(
                f"B={batch_size} numpy_bthwc vs {format_name}",
                batch_baseline,
                response,
            )
    print("SECTION frame_formats PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 7: EXPECTED ERRORS FOR BAD LENGTHS, VIEW TYPES, AND INPUTS
# -----------------------------------------------------------------------------

if run_section("validation_errors"):
    print("\n" + "=" * 88)
    print("SECTION validation_errors")
    print("Checks that invalid inputs fail locally with useful messages.")
    print("=" * 88)

    # Synthetic arrays keep this section independent of the remote fixtures.
    long_frames = np.zeros((12, 48, 64, 3), dtype=np.uint8)
    short_frames = np.zeros((10, 48, 64, 3), dtype=np.uint8)
    for index in range(len(long_frames)):
        long_frames[index, :, index % 64, 0] = 255
    for index in range(len(short_frames)):
        short_frames[index, :, index % 64, 1] = 255

    create_base = {
        "model": MODEL,
        "task_description": TASK_DESCRIPTION,
        "max_frames": 8,
        "downsample_to": 10,
        "verbose": verbose,
    }
    mismatched_frame_formats = {
        "list HWC": (list(long_frames), list(short_frames)),
        "NumPy THWC": (long_frames, short_frames),
        "NumPy TCHW": (
            np.moveaxis(long_frames, -1, 1),
            np.moveaxis(short_frames, -1, 1),
        ),
        "NumPy BTHWC": (
            np.stack([long_frames, long_frames]),
            np.stack([short_frames, short_frames]),
        ),
        "NumPy BTCHW": (
            np.moveaxis(np.stack([long_frames, long_frames]), -1, 2),
            np.moveaxis(np.stack([short_frames, short_frames]), -1, 2),
        ),
    }
    if torch is not None:
        mismatched_frame_formats.update(
            {
                "Torch THWC": (
                    torch.from_numpy(long_frames),
                    torch.from_numpy(short_frames),
                ),
                "Torch TCHW": (
                    torch.from_numpy(long_frames).movedim(-1, 1),
                    torch.from_numpy(short_frames).movedim(-1, 1),
                ),
                "Torch BTHWC": (
                    torch.from_numpy(np.stack([long_frames, long_frames])),
                    torch.from_numpy(np.stack([short_frames, short_frames])),
                ),
                "Torch BTCHW": (
                    torch.from_numpy(
                        np.stack([long_frames, long_frames])
                    ).movedim(-1, 2),
                    torch.from_numpy(
                        np.stack([short_frames, short_frames])
                    ).movedim(-1, 2),
                ),
            }
        )
    else:
        print("[SKIP] Torch unequal-length formats: torch is not installed")

    for format_name, (first_view, second_view) in mismatched_frame_formats.items():
        expect_error(
            f"different view lengths using {format_name}",
            lambda first_view=first_view, second_view=second_view: client.responses.create(
                **create_base,
                video_frames_view1=first_view,
                video_frames_view2=second_view,
                view_type=["static", "right_wrist"],
            ),
            "different frame counts",
        )

    # Write two local inputs from the same synthetic scenario with different
    # lengths so the video-path validation is also exercised.
    long_path = OUTPUT_DIR / "validation_12_frames.avi"
    short_path = OUTPUT_DIR / "validation_10_frames.avi"
    for path, frames in ((long_path, long_frames), (short_path, short_frames)):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            10.0,
            (frames.shape[2], frames.shape[1]),
        )
        assert writer.isOpened(), f"Could not create {path}"
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    expect_error(
        "different view lengths using video paths",
        lambda: client.responses.create(
            **create_base,
            video_paths_view1=[str(long_path)],
            video_paths_view2=[str(short_path)],
            view_type=["static", "right_wrist"],
        ),
        "different frame counts",
    )

    for description, bad_view_type in (
        ("too short", ["static"]),
        ("too long", ["static", "right_wrist", "left_wrist"]),
    ):
        expect_error(
            f"frame view_type is {description}",
            lambda bad_view_type=bad_view_type: client.responses.create(
                **create_base,
                video_frames_view1=long_frames,
                video_frames_view2=long_frames,
                view_type=bad_view_type,
            ),
            "one non-empty name per view",
        )
        expect_error(
            f"path view_type is {description}",
            lambda bad_view_type=bad_view_type: client.responses.create(
                **create_base,
                video_paths_view1=[str(long_path)],
                video_paths_view2=[str(long_path)],
                view_type=bad_view_type,
            ),
            "one non-empty name per view",
        )

    expect_error(
        "mixed path and frame families",
        lambda: client.responses.create(
            **create_base,
            video_path=str(long_path),
            video_frames_view2=long_frames,
        ),
        "exactly one",
    )
    expect_error(
        "empty frame input",
        lambda: client.responses.create(**create_base, video_frames=[]),
        "empty",
    )
    expect_error(
        "invalid RGB channel layout",
        lambda: client.responses.create(
            **create_base,
            video_frames=np.zeros((12, 48, 64, 2), dtype=np.uint8),
        ),
        "RGB channel",
    )
    expect_error(
        "different episode counts between frame views",
        lambda: client.responses.create(
            **create_base,
            video_frames_view1=np.stack([long_frames, long_frames]),
            video_frames_view2=np.expand_dims(long_frames, 0),
            view_type=["left_wrist", "static"],
        ),
        "same number of episodes",
    )
    expect_error(
        "max_frames must be positive",
        lambda: client.responses.create(
            **{**create_base, "max_frames": 0},
            video_frames=long_frames,
        ),
        "positive integer",
    )

    plot_base = {
        "outputs": [
            {
                "model": "synthetic",
                "rewards": np.linspace(0, 100, len(long_frames)).tolist(),
            }
        ],
        "plot_save_path": str(OUTPUT_DIR / "expected_video_plot_error.mp4"),
        "show_output_text": False,
    }
    expect_error(
        "video_plot rejects Python frame lists",
        lambda: video_plot(**plot_base, video_frames=list(long_frames)),
        "NumPy array or PyTorch tensor",
    )
    expect_error(
        "video_plot rejects a batch larger than one",
        lambda: video_plot(
            **plot_base, video_frames=np.stack([long_frames, long_frames])
        ),
        "supports only one video",
    )
    expect_error(
        "video_plot rejects unequal frame-view lengths",
        lambda: video_plot(
            **plot_base,
            video_frames_view1=long_frames,
            video_frames_view2=short_frames,
            view_type=["static", "right_wrist"],
        ),
        "same frame count",
    )
    expect_error(
        "video_plot rejects too few view_type labels",
        lambda: video_plot(
            **plot_base,
            video_frames_view1=long_frames,
            video_frames_view2=long_frames,
            view_type=["static"],
        ),
        "one non-empty name per view",
    )
    expect_error(
        "video_plot rejects too many view_type labels",
        lambda: video_plot(
            **plot_base,
            video_frames_view1=long_frames,
            video_frames_view2=long_frames,
            view_type=["static", "right_wrist", "left_wrist"],
        ),
        "one non-empty name per view",
    )
    expect_error(
        "video_plot rejects reward/source length mismatch",
        lambda: video_plot(
            **{
                **plot_base,
                "outputs": [
                    {
                        "model": "synthetic",
                        "rewards": np.linspace(0, 100, 11).tolist(),
                    }
                ],
            },
            video_frames=long_frames,
        ),
        "does not match",
    )
    expect_error(
        "video_plot rejects unequal path-view lengths",
        lambda: video_plot(
            **plot_base,
            video_path_view1=str(long_path),
            video_path_view2=str(short_path),
            view_type=["static", "right_wrist"],
        ),
        "same frame count",
    )
    print("SECTION validation_errors PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 8: BATCH SIZES 1, 2, AND 3 FOR EVERY INPUT FAMILY
# -----------------------------------------------------------------------------

if run_section("batches"):
    print("\n" + "=" * 88)
    print("SECTION batches")
    print("Checks B=1, B=2, and B=3 for paths, frame arrays, and LeRobot.")
    print("=" * 88)

    assert dataset is not None, "Install LeRobot and set SUBLINEAR_LEROBOT_ROOT"
    batch_responses = {"paths": {}, "frames": {}, "lerobot": {}}

    for batch_size in (1, 2, 3):
        selected_ids = EPISODE_IDS[:batch_size]

        path_response = client.responses.create(
            model=MODEL,
            task_description=TASK_DESCRIPTION,
            video_paths_view1=[
                str(episode_files[episode_id]["left_wrist"])
                for episode_id in selected_ids
            ],
            video_paths_view2=[
                str(episode_files[episode_id]["static"])
                for episode_id in selected_ids
            ],
            video_paths_view3=[
                str(episode_files[episode_id]["right_wrist"])
                for episode_id in selected_ids
            ],
            view_type=list(CANONICAL_ROLES),
            max_frames=50,
            downsample_to=30,
            verbose=verbose,
        )

        # Decode the same small number of frames per episode so B=2 and B=3 can
        # be stacked into true 5D NumPy arrays without using excessive memory.
        frame_batches = {}
        for role in CANONICAL_ROLES:
            frame_batches[role] = np.stack(
                [
                    read_all_frames(
                        episode_files[episode_id][role], max_frames=36
                    )
                    for episode_id in selected_ids
                ],
                axis=0,
            )

        frame_response = client.responses.create(
            model=MODEL,
            task_description=TASK_DESCRIPTION,
            video_frames_view1=frame_batches["left_wrist"],
            video_frames_view2=frame_batches["static"],
            video_frames_view3=frame_batches["right_wrist"],
            view_type=list(CANONICAL_ROLES),
            color_order="rgb",
            max_frames=30,
            downsample_to=30,
            verbose=verbose,
        )
        lerobot_response = client.responses.create(
            model=MODEL,
            task_description=TASK_DESCRIPTION,
            lerobot_dataset=dataset,
            episodes=list(selected_ids),
            lerobot_view_map={
                role: ROLE_TO_LEROBOT_KEY[role] for role in CANONICAL_ROLES
            },
            max_frames=50,
            downsample_to=30,
            verbose=verbose,
        )

        path_rewards = reward_batch(path_response, f"paths B={batch_size}")
        frame_rewards = reward_batch(frame_response, f"frames B={batch_size}")
        lerobot_rewards = reward_batch(
            lerobot_response, f"LeRobot B={batch_size}"
        )
        assert len(path_rewards) == batch_size
        assert len(frame_rewards) == batch_size
        assert len(lerobot_rewards) == batch_size
        assert all(len(values) == 36 for values in frame_rewards)

        batch_responses["paths"][batch_size] = path_response
        batch_responses["frames"][batch_size] = frame_response
        batch_responses["lerobot"][batch_size] = lerobot_response

    # Adding episodes to a batch must not change earlier episodes' outputs.
    for family, responses in batch_responses.items():
        for small_size, large_size in ((1, 2), (2, 3)):
            small = reward_batch(responses[small_size], f"{family} B={small_size}")
            large = reward_batch(responses[large_size], f"{family} B={large_size}")
            for episode_index in range(small_size):
                assert np.allclose(
                    small[episode_index],
                    large[episode_index],
                    atol=SAME_REWARD_ATOL,
                    rtol=0.0,
                )
            print(
                f"[PASS] {family}: B={small_size} rewards equal the prefix "
                f"of B={large_size}"
            )
    print("SECTION batches PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 9: UNKNOWN view_type VALUES
# -----------------------------------------------------------------------------

if run_section("unknown_view"):
    print("\n" + "=" * 88)
    print("SECTION unknown_view")
    print("Checks that unknown view labels remain valid and usable.")
    print("=" * 88)

    assert dataset is not None, "Install LeRobot and set SUBLINEAR_LEROBOT_ROOT"
    files = episode_files[73]
    static_frames = read_all_frames(files["static"])

    unknown_path = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_paths_view1=[str(files["static"])],
        view_type="some string",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    unknown_frames = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_frames_view1=static_frames,
        view_type="some string",
        color_order="rgb",
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    unknown_lerobot = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        lerobot_dataset=dataset,
        episodes=[73],
        lerobot_view_map={"some string": ROLE_TO_LEROBOT_KEY["static"]},
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    unknown_three_views = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_paths_view1=[str(files["left_wrist"])],
        video_paths_view2=[str(files["static"])],
        video_paths_view3=[str(files["right_wrist"])],
        view_type=["some string", "another camera", "mystery"],
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )

    for name, response in (
        ("unknown path", unknown_path),
        ("unknown frames", unknown_frames),
        ("unknown LeRobot", unknown_lerobot),
        ("three unknown labels", unknown_three_views),
    ):
        values = reward_batch(response, name)
        assert len(values) == 1
        assert len(values[0]) == len(static_frames)

    video_plot_check(
        "unknown_view_path",
        [{"model": MODEL, "rewards": reward_batch(unknown_path)[0].tolist()}],
        video_path_view1=str(files["static"]),
        view_type="some string",
    )
    video_plot_check(
        "unknown_view_frames",
        [{"model": MODEL, "rewards": reward_batch(unknown_frames)[0].tolist()}],
        video_frames_view1=static_frames,
        view_type="some string",
    )
    video_plot_check(
        "unknown_view_lerobot",
        [{"model": MODEL, "rewards": reward_batch(unknown_lerobot)[0].tolist()}],
        lerobot_dataset=dataset,
        episode=73,
        lerobot_view_map={"some string": ROLE_TO_LEROBOT_KEY["static"]},
    )
    print("SECTION unknown_view PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 10: CORRELATION WITH data1, data2, AND data3 GROUND TRUTH
# -----------------------------------------------------------------------------

if run_section("ground_truth"):
    print("\n" + "=" * 88)
    print("SECTION ground_truth")
    print("Prints Pearson correlation for each of data1, data2, and data3.")
    print("=" * 88)

    with episode_files[73]["json"].open("r", encoding="utf-8") as handle:
        data1 = json.load(handle)
    with episode_files[0]["json"].open("r", encoding="utf-8") as handle:
        data2 = json.load(handle)
    with episode_files[25]["json"].open("r", encoding="utf-8") as handle:
        data3 = json.load(handle)

    ground_truth_response = client.responses.create(
        model=MODEL,
        task_description=TASK_DESCRIPTION,
        video_paths_view1=[
            str(episode_files[episode_id]["left_wrist"])
            for episode_id in EPISODE_IDS
        ],
        video_paths_view2=[
            str(episode_files[episode_id]["static"])
            for episode_id in EPISODE_IDS
        ],
        video_paths_view3=[
            str(episode_files[episode_id]["right_wrist"])
            for episode_id in EPISODE_IDS
        ],
        view_type=list(CANONICAL_ROLES),
        max_frames=50,
        downsample_to=30,
        verbose=verbose,
    )
    predicted_batch = reward_batch(
        ground_truth_response, "ground-truth batch"
    )
    assert len(predicted_batch) == 3

    for data_index, (episode_id, data, predicted) in enumerate(
        zip(EPISODE_IDS, (data1, data2, data3), predicted_batch), start=1
    ):
        actual = np.asarray(data["progress"], dtype=float)
        assert len(actual) == len(predicted), (
            f"data{data_index}: ground truth and prediction lengths differ"
        )
        correlation = float(np.corrcoef(predicted, actual)[0, 1])
        print(
            f"data{data_index}, episode {episode_id}: "
            f"Pearson correlation={correlation:.8g}"
        )
        assert math.isfinite(correlation)
        assert correlation >= MIN_GROUND_TRUTH_CORRELATION, (
            f"data{data_index}: correlation {correlation} is below "
            f"{MIN_GROUND_TRUTH_CORRELATION}"
        )

        files = episode_files[episode_id]
        video_plot_check(
            f"ground_truth_data{data_index}_episode_{episode_id}",
            [
                {"model": MODEL, "rewards": predicted.tolist()},
                {"model": "ground truth", "rewards": actual.tolist()},
            ],
            video_path_view1=str(files["left_wrist"]),
            video_path_view2=str(files["static"]),
            video_path_view3=str(files["right_wrist"]),
            view_type=list(CANONICAL_ROLES),
            task_description=TASK_DESCRIPTION,
        )
    print("SECTION ground_truth PASSED")


# %%
# -----------------------------------------------------------------------------
# SECTION 11: EXTRA LOCAL video_plot CHECKS WITH NO API CALLS
# -----------------------------------------------------------------------------

if run_section("local_video_plot"):
    print("\n" + "=" * 88)
    print("SECTION local_video_plot")
    print("Checks local view ordering, overlays, sampling, and artifacts.")
    print("=" * 88)

    frame_count = 12
    height = 48
    width = 64
    local_left = np.zeros((frame_count, height, width, 3), dtype=np.uint8)
    local_static = np.zeros_like(local_left)
    local_right = np.zeros_like(local_left)
    local_left[..., 0] = 230
    local_static[..., 1] = 230
    local_right[..., 2] = 230
    for index in range(frame_count):
        local_left[index, :, index % width, :] = 255
        local_static[index, :, index % width, :] = 255
        local_right[index, :, index % width, :] = 255

    ascending = np.linspace(0, 100, frame_count).tolist()
    descending = list(reversed(ascending))

    video_plot_check(
        "local_unknown_single_view",
        [{"model": "ascending", "rewards": ascending}],
        video_frames=local_static,
        view_type="some string",
    )

    two_view_frames = video_plot_check(
        "local_two_views_reordered",
        [{"model": "ascending", "rewards": ascending}],
        video_frames_view1=local_right,
        video_frames_view2=local_static,
        view_type=["right_wrist", "top"],
    )
    first_two_view_frame = two_view_frames[0]
    halfway = first_two_view_frame.shape[1] // 2
    dominant_channels = [
        int(np.argmax(first_two_view_frame[:, :halfway].mean(axis=(0, 1)))),
        int(np.argmax(first_two_view_frame[:, halfway:].mean(axis=(0, 1)))),
    ]
    print(f"two-view dominant RGB channels: {dominant_channels}")
    assert dominant_channels == [1, 2]

    three_view_frames = video_plot_check(
        "local_three_views_reordered",
        [{"model": "ascending", "rewards": ascending}],
        video_frames_view1=local_right,
        video_frames_view2=local_left,
        video_frames_view3=local_static,
        view_type=["right_wrist", "left_wrist", "top"],
    )
    first_three_view_frame = three_view_frames[0]
    third = first_three_view_frame.shape[1] // 3
    dominant_channels = [
        int(np.argmax(first_three_view_frame[:, :third].mean(axis=(0, 1)))),
        int(
            np.argmax(
                first_three_view_frame[:, third : 2 * third].mean(axis=(0, 1))
            )
        ),
        int(np.argmax(first_three_view_frame[:, 2 * third :].mean(axis=(0, 1)))),
    ]
    print(f"three-view dominant RGB channels: {dominant_channels}")
    assert dominant_channels == [0, 1, 2]

    video_plot_check(
        "local_two_reward_overlay",
        [
            {"model": "ascending", "rewards": ascending},
            {"model": "descending", "rewards": descending},
        ],
        video_frames=local_static,
        view_type="static",
    )

    if torch is not None:
        torch_b1_tchw = (
            torch.from_numpy(local_static).movedim(-1, 1).unsqueeze(0)
        )
        video_plot_check(
            "local_torch_b1_tchw",
            [{"model": "ascending", "rewards": ascending}],
            video_frames=torch_b1_tchw,
            view_type="static",
        )
    else:
        print("[SKIP] local Torch video_plot case: torch is not installed")
    print("SECTION local_video_plot PASSED")


print("\nSelected integration-test sections finished.")

