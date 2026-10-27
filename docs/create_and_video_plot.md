# `Sublinear.create()` and `video_plot()`

This document defines the supported input contracts, batching rules, return values, and plotting behavior of the Sublinear Python SDK.

## Installation and authentication

```bash
pip install -U sublinear
export SUBLINEAR_API_KEY="sk_..."
```

For LeRobot:

```bash
pip install -U "sublinear[lerobot]"
```

```python
import os
from sublinear import Sublinear, video_plot

client = Sublinear(
    api_key=os.environ["SUBLINEAR_API_KEY"]
)
```

`client.create(...)` and `client.responses.create(...)` currently call the same method.

## `Sublinear`

```python
Sublinear(api_key: str)
```

- `api_key` is required and becomes a bearer token.
- Keep API keys in environment variables.

## `Sublinear.create()`

Current signature:

```python
client.responses.create(
    *,
    model: str,
    task_description: str,
    video_path=None,
    video_paths=None,
    video_paths_view1=None,
    video_paths_view2=None,
    video_paths_view3=None,
    video_frames=None,
    video_frames_view1=None,
    video_frames_view2=None,
    video_frames_view3=None,
    view_type=None,
    lerobot_dataset=None,
    episodes=None,
    lerobot_view_map=None,
    color_order="rgb",
    fps=None,
    max_frames=50,
    max_videos=30,
    ffmpeg_path=None,
    ffprobe_path=None,
    lerobot_remux_fallback="transcode",
    verbose=False,
    **input_json,
)
```

The call is synchronous. It:

1. Validates and packages the selected episodes.
2. Uploads a temporary bundle.
3. Starts an inference job.
4. Polls until completion.
5. Returns the result as an attribute-accessible object.

Interrupting the wait with `Ctrl-C` requests job cancellation.


### Required arguments

| Argument | Meaning |
|---|---|
| `model` | Backend model identifier. |
| `task_description` | Non-empty natural-language description of the desired task. |

### Choose exactly one input family

| Family | Arguments |
|---|---|
| LeRobot | `lerobot_dataset`, optionally `episodes` and `lerobot_view_map` |
| Generic frames | `video_frames` |
| Numbered frame views | `video_frames_view1`, `video_frames_view2`, `video_frames_view3` |
| Generic path | `video_path` or `video_paths` |
| Numbered path views | `video_paths_view1`, `video_paths_view2`, `video_paths_view3` |

Do not combine path, frame, and LeRobot families. Do not combine a generic source with numbered sources.


## LeRobot input

`lerobot_dataset` must be an already-loaded, local `LeRobotDataset` with video files available on disk.

```python
from lerobot.datasets import LeRobotDataset

episode_ids = [73, 0, 25]
dataset = LeRobotDataset(
    repo_id="DreamMachines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_full",
    episodes=episode_ids,
    download_videos=True,
)

response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    lerobot_dataset=dataset,
    episodes=episode_ids,
    lerobot_view_map={
        "left_wrist": "observation.images.left_wrist",
        "static": "observation.images.top",
        "right_wrist": "observation.images.right_wrist",
    },
)
```

`lerobot_view_map` maps semantic roles to LeRobot feature names:

```text
role -> dataset feature
```

Sublinear automatically looks for these defaults when the map is omitted:

```python
{
    "static": "observation.images.top",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}
```

Explicit maps are recommended.

The fast LeRobot path requires:

- MP4-backed video features.
- RGB rather than depth-video features.
- Valid FPS metadata.
- Local video files.
- No `image_transforms`.
- No video `delta_timestamps`.

It rejects image-backed camera features because it packages compressed videos without iterating through frame Parquet.

### `episodes`

`episodes` may be:

```python
episodes=7
episodes=[7, 2, 19]
```

The result follows the requested order.

If `episodes=None`:

- The dataset's preselected `dataset.episodes` is used when available.
- Otherwise all episodes are selected only when their count is within `max_videos`.
- Large datasets should always pass explicit episode IDs.

Duplicate, negative, unavailable, and excessive episode selections fail locally.


### Local video paths

One episode:

```python
response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    video_path="/videos/episode.mp4",
)
```

Multiple episodes:

```python
response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    video_paths=[
        "/videos/episode_000.mp4",
        "/videos/episode_001.mp4",
    ],
)
```

`video_paths` is a one-view batch. Result ordering follows path ordering:

```python
response.rewards[0]  # episode_000.mp4
response.rewards[1]  # episode_001.mp4
```

### Separate path views

```python
response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    video_paths_view1=[
        "/videos/000_left_wrist.mp4",
        "/videos/001_left_wrist.mp4",
    ],
    video_paths_view2=[
        "/videos/000_static.mp4",
        "/videos/001_static.mp4",
    ],
    video_paths_view3=[
        "/videos/000_right_wrist.mp4",
        "/videos/001_right_wrist.mp4",
    ],
    view_type=[
        "left_wrist",
        "static",
        "right_wrist",
    ],
)
```

For numbered view batches:

- Every view must contain the same number of episodes.
- Position `i` across the view lists defines one episode.
- All views belonging to an episode must have the same decodable frame count.
- A numbered argument may also receive one path instead of a list.

### In-memory frames

Supported `create()` layouts are:

```text
(T, H, W, 3)      one channel-last video
(T, 3, H, W)      one channel-first video
(B, T, H, W, 3)   channel-last batch
(B, T, 3, H, W)   channel-first batch
```

Inputs may be NumPy arrays or Torch tensors. `create()` also accepts a list of same-shaped frames or a list of 4D episode arrays.

Channel count must be three. For floating-point input:

- Values entirely within `[0, 1]` are multiplied by 255.
- Other values are clipped to `[0, 255]`.
- Encoded frames use `uint8`.

One frame-array episode:

```python
response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    video_frames=frames_thwc,
    color_order="rgb",
    fps=30,
)
```

Three frame views:

```python
response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    video_frames_view1=left_frames,
    video_frames_view2=static_frames,
    video_frames_view3=right_frames,
    view_type=[
        "left_wrist",
        "static",
        "right_wrist",
    ],
    color_order="rgb",
    fps=30,
)
```

### `color_order`

`color_order` applies to frame inputs:

```python
color_order="rgb"
color_order="bgr"
color_order=["rgb", "rgb", "bgr"]
```

A sequence must contain exactly one value per view.

Path inputs and LeRobot MP4 inputs determine colors while decoding and do not use this option.

### View roles and ordering

For three recognized views, Sublinear canonicalizes the horizontal order to:

```text
left_wrist | static | right_wrist
```

Recognized static synonyms include:

```text
top, static, external, side
```

For two recognized views, the order is:

```text
static | wrist
```

Unknown, incomplete, duplicate, or ambiguous role classifications retain their original `view1`, `view2`, `view3` order.

For predictable behavior, use canonical roles:

```python
view_type=[
    "left_wrist",
    "static",
    "right_wrist",
]
```

When using an already-stitched video, pass it as one generic source:

```python
response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    video_path="/videos/already_stitched.mp4",
)
```

Do not pass the same stitched source as three numbered views.


### FFmpeg controls

| Argument | Meaning |
|---|---|
| `ffmpeg_path` | Optional explicit FFmpeg executable. |
| `ffprobe_path` | Optional explicit FFprobe executable. |
| `lerobot_remux_fallback` | `"transcode"` or `"error"`. |

The SDK tries to copy or remux compressed episode clips. When an exact remux is unsafe, the default `"transcode"` fallback re-encodes the clip. Use `"error"` when an unexpected transcode must fail instead.

For LeRobot, an explicit `fps` must equal the dataset metadata FPS.

## Sampling and batching

### `max_frames`

```python
max_frames=50
max_frames=None
```

`max_frames` is the maximum number of frames packaged per source episode.

When sampling is required, the SDK chooses uniformly spaced indices including the first and final frames. `None` disables this client-side limit.

This does not shorten the returned reward sequence. The backend interpolates model values back to the original source timeline:

```python
len(response.rewards[i]) == (
    response.frame_mapping[i]["source_frame_count"]
)
```

### `downsample_to`

`downsample_to` is currently passed through `**input_json`:

```python
response = client.responses.create(
    ...,
    downsample_to=30,
)
```

It controls the number of model evaluation timesteps after client-side packaging. The backend currently accepts integers from 1 through 100.

A typical request uses:

```python
max_frames=50
downsample_to=30
```

Conceptually:

```text
original frames
    -> at most max_frames packaged frames
    -> at most downsample_to model positions
    -> interpolation back to original frame count
```

### `max_videos`

`max_videos` is the maximum episode count in one job:

```python
max_videos=30
```

The default is 30. The client permits values from 1 through 100, but model-specific batch limits can be lower. Exceeding the selected limit raises an error; inputs are not silently truncated.

Split larger workloads into deterministic batches and retain the requested episode order.

### `fps`

For path inputs:

- The SDK detects source FPS when `fps=None`.
- An explicit value overrides the detected value.

For frame inputs:

- The default is 30 FPS.
- `fps` describes the encoded temporary video.

For LeRobot:

- Dataset metadata supplies FPS.
- An explicit value must match it.

## Create response

The current result exposes:

| Field | Meaning |
|---|---|
| `model` | Model identifier. |
| `rewards` | Full-length per-frame rewards grouped by episode. |
| `output_text` | Optional full-length text/trace entries grouped by episode. |
| `success_probs` | Optional full-length success probabilities. |
| `success` | Optional episode-level success result. |
| `frame_mapping` | Sampling and source-index metadata per episode. |
| `input_file_size_bytes` | Uploaded bundle size. |

Always treat the outer `rewards` dimension as the episode dimension:

```python
for episode_id, rewards in zip(
    episode_ids,
    response.rewards,
):
    ...
```

`frame_mapping[i]` currently contains:

| Field | Meaning |
|---|---|
| `episode_index` | Source episode ID or batch position. |
| `dataset_from_index` | LeRobot global starting index when available. |
| `source_frame_count` | Original episode frame count. |
| `sampled_frame_indices` | Frames packaged by the client, or `None` if unsampled. |
| `rewardgen_frame_positions` | Model positions within the packaged video. |
| `rewardgen_source_frame_indices` | Corresponding positions in the original video. |

## `video_plot()`

Current signature:

```python
video_plot(
    outputs,
    plot_save_path,
    video_path=None,
    video_path_view1=None,
    video_path_view2=None,
    video_path_view3=None,
    video_frames=None,
    video_frames_view1=None,
    video_frames_view2=None,
    video_frames_view3=None,
    view_type=None,
    lerobot_dataset=None,
    episode=None,
    lerobot_view_map=None,
    max_frames=50,
    task_description=None,
    show_all_frames=False,
    fps=2,
    wrap_width=26,
    font_scale=1,
    font_height=30,
    text_thickness=2,
    line_type=2,
    show_output_text=True,
    cfg=None,
    env_rew_lab="Ground-truth reward",
    save_json=True,
    verbose=False,
    write_json=True,
    x_axis_label="Timestep",
    y_axis_label="Predicted task progress (%)",
    return_frames=False,
    target_video_height=768,
    plot_line_width=5,
    show_legend=True,
    legend_font_size=None,
    x_axis_label_size=None,
    y_axis_label_size=None,
    x_axis_tick_size=None,
    y_axis_tick_size=None,
    plot_width_relative_to_video=0.5,
)
```

### Plot source rules

`video_plot()` renders one episode per call.

Choose exactly one source family:

| Family | Arguments |
|---|---|
| Generic episode | `video_path` or `video_frames` |
| Numbered episode views | `video_path_view1..3` or `video_frames_view1..3` |
| LeRobot episode | `lerobot_dataset` and singular `episode` |

The generic source may already contain horizontally stitched views.

Numbered views are synchronized and horizontally stitched before rendering. Each numbered view may be a path or frame input, but a single view cannot receive both.

For frame inputs, use one NumPy array or Torch tensor with:

```text
(T, H, W, 3)
(T, 3, H, W)
(1, T, H, W, 3)
(1, T, 3, H, W)
```

A 5D batch must have `B=1`.

The current implementation contains Python-list normalization, but the integration test currently marks Python frame lists unsupported. Until that inconsistency is resolved, convert lists explicitly:

```python
video_frames=np.asarray(frame_list)
```

`video_plot()` assumes frame arrays are RGB and has no `color_order` argument.

### `outputs`

`outputs` is a non-empty list of one to seven dictionaries:

```python
outputs=[
    {
        "model": "dm-1026",
        "rewards": rewards,
        "plot_color": "darkblue",
        "output_text": output_text,
        "success_probs": success_probs,
    }
]
```

Required keys:

| Key | Meaning |
|---|---|
| `model` | Legend label. |
| `rewards` | One full-length reward sequence. |

Optional keys:

| Key | Meaning |
|---|---|
| `plot_color` | Matplotlib-compatible color. |
| `output_text` | Per-frame text or reasoning trace. |
| `success_probs` | Preserved in JSON but not plotted as a separate line. |

All outputs must:

- Have the same reward length.
- Refer to the same episode.
- Match the original source frame count.

Convert NumPy arrays to lists when JSON output is enabled:

```python
{
    "model": "comparison",
    "rewards": reward_array.tolist(),
}
```

A singleton response batch is accepted, but explicit indexing is clearer:

```python
response.rewards[0]
```

### Multiple curves

```python
video_plot(
    outputs=[
        {
            "model": "Sublinear",
            "rewards": predicted_rewards,
            "plot_color": "darkblue",
        },
        {
            "model": "Ground truth",
            "rewards": ground_truth_rewards,
            "plot_color": "darkred",
        },
    ],
    plot_save_path="./comparison.mp4",
    video_path="./episode.mp4",
    show_output_text=False,
)
```

At most seven curves are currently supported.

### LeRobot plot

```python
video_plot(
    outputs=[
        {
            "model": response.model,
            "rewards": response.rewards[0],
        }
    ],
    plot_save_path="./episode_7.mp4",
    lerobot_dataset=dataset,
    episode=7,
    lerobot_view_map={
        "left_wrist": "observation.images.left_wrist",
        "static": "observation.images.top",
        "right_wrist": "observation.images.right_wrist",
    },
)
```

Use singular `episode`, not `episodes`.

### Plot sampling

`max_frames` limits rendered frames:

```python
max_frames=50
```

The plotter selects evenly spaced source indices and selects rewards at those same indices. It still requires a full-length input reward sequence.

`fps` controls output playback speed:

```python
fps=5
```

It does not describe source-video FPS and does not change reward alignment.

### Output text

Text is displayed only when exactly one plotted output supplies `output_text`.

With:

```python
show_output_text=True
show_all_frames=False
```

the rendered video keeps frame zero and frames containing non-empty text.

Use:

```python
show_all_frames=True
```

to retain all sampled frames while showing text.

For reward comparisons, prefer:

```python
show_output_text=False
```

### Artifacts and return value

Use an `.mp4` destination:

```python
plot_save_path="./episode.mp4"
```

The current implementation writes a temporary MJPG AVI, transcodes it to H.264 MP4, and removes the AVI.

When both flags are true:

```python
save_json=True
write_json=True
```

it also writes:

```text
episode.json
```

The JSON contains:

```json
{
  "outputs": [],
  "meta": {
    "source_frame_count": 0,
    "sampled_frame_indices": []
  }
}
```

With:

```python
return_frames=True
```

the function returns the sampled, stitched RGB frames. Otherwise it returns `None`.

### Plot formatting arguments

| Argument | Effect |
|---|---|
| `task_description` | Adds a task header above the video. |
| `plot_width_relative_to_video` | Plot width as a fraction of displayed video width; must be positive. |
| `target_video_height` | Target video-panel height. |
| `plot_line_width` | Reward-curve width. |
| `show_legend` | Shows or hides the legend. |
| `legend_font_size` | Explicit Matplotlib legend size. |
| `x_axis_label`, `y_axis_label` | Axis text. |
| `x_axis_label_size`, `y_axis_label_size` | Axis-label sizes. |
| `x_axis_tick_size`, `y_axis_tick_size` | Tick-label sizes. |
| `wrap_width` | Approximate output-text wrapping width. |
| `font_scale`, `font_height` | Output-text sizing. |
| `text_thickness`, `line_type` | OpenCV text rendering. |
| `verbose` | Prints frame-layout information. |

`cfg` and `env_rew_lab` are accepted legacy arguments but are not used by the current implementation.

## End-to-end batched-create/single-plot pattern

```python
episode_ids = [7, 2, 19]

response = client.responses.create(
    model="dm-1026",
    task_description="Take an actuator from the box and place it in the tray",
    lerobot_dataset=dataset,
    episodes=episode_ids,
    lerobot_view_map=view_map,
    max_frames=50,
    downsample_to=30,
)

for response_index, episode_id in enumerate(episode_ids):
    rewards = response.rewards[response_index]

    assert len(rewards) == (
        response.frame_mapping[response_index][
            "source_frame_count"
        ]
    )

    video_plot(
        outputs=[
            {
                "model": response.model,
                "rewards": rewards,
            }
        ],
        plot_save_path=f"./episode_{episode_id}.mp4",
        lerobot_dataset=dataset,
        episode=episode_id,
        lerobot_view_map=view_map,
        show_output_text=False,
    )
```

This is the safest pattern for coding agents:

```text
batch with create()
preserve requested episode order
index one response episode
plot one source episode
```

## Common errors

### More than one input family

```text
Provide exactly one of video paths, video frames, or lerobot_dataset.
```

Remove all input arguments outside the intended family.

### Different view counts

Every numbered view must contain the same episode count for `create()` and the same frame count for each episode.

### Ambiguous frame shape

A dimension layout is ambiguous when both the expected channel dimension and the last dimension equal three. Convert explicitly to THWC or TCHW with an unambiguous spatial shape.

### Reward/source mismatch

`video_plot()` requires:

```python
len(rewards) == original_source_frame_count
```

Do not pass only inference-anchor rewards.

### Hub LeRobot dataset has no videos

Recreate the dataset with:

```python
download_videos=True
```

### JSON serialization failure

Use plain Python values:

```python
np.asarray(rewards, dtype=float).tolist()
```

### Large dataset selection

Pass explicit `episodes=` and divide work into batches no larger than `max_videos`.

