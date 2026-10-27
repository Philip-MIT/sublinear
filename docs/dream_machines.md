## Installation

The sublinear python package is actively being updated so please ensure you use the most updated version.

```bash
pip install -U --no-cache-dir sublinear
```

## Notes for Dream Machines
- Model is fine-tuned on the cropped 224x224 frames, with all 3 views always available (left_wrist, top, right_wrist)
- Model seems to perform best when predicting rewards at 30 timesteps per video (downsample_to=30)
- By default, client submits a maximum of 30 videos (max_videos=30) to the model to be processed in parallel with batched inference
    - can try larger batch sizes, but may lead to crashing
- More detailed docs at: [/data/sls/scratch/pschro/sl/create_and_video_plot.md](https://github.com/Philip-MIT/sublinear/blob/main/docs/create_and_video_plot.md)

## Compute and plot rewards for local lerobot dataset

Create an API key from the Sublinear dashboard (https://sublinear.dev/login), then submit a local lerobot dataset:

```python
from sublinear import Sublinear, video_plot
from lerobot.datasets import LeRobotDataset

# API_KEY="sk_...."
# DATA_DIR = '/path/to/data'
# PLOT_OUTPUT_DIR = '/path/to/plots'

client = Sublinear(api_key=API_KEY)

episode_ids = [73, 0, 25]
dataset = LeRobotDataset(
    repo_id="local/actuator_unboxing_fullbox_eval_pi07_padmask_20261001", 
    root=f"{DATA_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_full"
)


response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    lerobot_dataset=dataset,
    episodes=episode_ids,
    lerobot_view_map={
        "left_wrist": "observation.images.left_wrist",
        "top": "observation.images.top",
        "right_wrist": "observation.images.right_wrist",
    },
    verbose=True
)

# Print rewards for each of the 3 episodes in episode_ids
print(response.rewards[0])
print(response.rewards[1])
print(response.rewards[2])

# Plot rewards for each episode
for i in range(len(episode_ids)):
    video_plot(
        outputs=[{"model": "dm-1026", "rewards": response.rewards[i]}],
        plot_save_path=f"{PLOT_OUTPUT_DIR}/video_plot_episode_{episode_ids[i]}.mp4",
        lerobot_dataset=dataset, 
        episode=episode_ids[i],
        lerobot_view_map={
            "left_wrist": "observation.images.left_wrist",
            "top": "observation.images.top",
            "right_wrist": "observation.images.right_wrist",
        },
        plot_width_relative_to_video=0.42, # works well when including 3 square views 
    )


```

## Episodes from dataset on hub

```python

dataset = LeRobotDataset(
    repo_id="DreamMachines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_full",
    episodes=episode_ids,
    revision="main",       # Prefer a pinned tag or commit
    download_videos=True,  # Required by Sublinear's compressed-video path
)


response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    lerobot_dataset=dataset,
    episodes=episode_ids,
    lerobot_view_map={
        "left_wrist": "observation.images.left_wrist",
        "top": "observation.images.top",
        "right_wrist": "observation.images.right_wrist",
    },
    verbose=True
)

for i in range(len(episode_ids)):
    video_plot(
        outputs=[{"model": "dm-1026", "rewards": response.rewards[i]}],
        plot_save_path=f"{PLOT_OUTPUT_DIR}/lerobot_episode_{episode_ids[i]}.mp4",
        lerobot_dataset=dataset, 
        episode=episode_ids[i],
        lerobot_view_map={
            "left_wrist": "observation.images.left_wrist",
            "top": "observation.images.top",
            "right_wrist": "observation.images.right_wrist",
        },
        plot_width_relative_to_video=0.42, # works well when including 3 square views 
    )


```

## Video paths separated across views
```python

# can use example_videos in this repo at https://github.com/Philip-MIT/sublinear/tree/main/example_videos/dream_machines

# VIDEO_DIR = '/path/to/example_videos/dream_machines'

video_paths_view1=[
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_left_wrist.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_left_wrist.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_left_wrist.mp4",
]
video_paths_view2=[
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_top.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_top.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_top.mp4",
]
video_paths_view3=[
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_right_wrist.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_right_wrist.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_right_wrist.mp4",
]


response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    video_paths_view1=video_paths_view1,
    video_paths_view2=video_paths_view2,
    video_paths_view3=video_paths_view3,
    view_type=["left_wrist", "top", "right_wrist"],
    verbose=True
)



video_plot(
    outputs=[{"model": "dm-1026", "rewards": response.rewards[video_idx]}],
    plot_save_path=f"/data/sls/scratch/pschro/sole/video_path_video_{video_idx}.mp4",
    video_path_view1=video_path_view1,
    video_path_view2=video_path_view2,
    video_path_view3=video_path_view3,
    view_type=["left_wrist", "top", "right_wrist"],
    plot_width_relative_to_video=0.42, # works well when including 3 square views 
    fps=5,
)

```

## Video frames separated across views

```python
from sublinear import load_video_frames

video_frames_view1 = []
for path in video_paths_view1:
    video_frames_view1.append(load_video_frames(path, max_frames=float('inf')))

video_frames_view2 = []
for path in video_paths_view2:
    video_frames_view2.append(load_video_frames(path, max_frames=float('inf')))

video_frames_view3 = []
for path in video_paths_view3:
    video_frames_view3.append(load_video_frames(path, max_frames=float('inf')))



# video frames can be numpy arrays, torch tensors, or lists (with frames still as numpy arrays)
response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    video_frames_view1=video_frames_view1,
    video_frames_view2=video_frames_view2,
    video_frames_view3=video_frames_view3,
    view_type=["left_wrist", "top", "right_wrist"], 
    verbose=True
)


# Plot rewards for each episode
for video_idx in range(len(video_frames_view1)):
    video_plot(
        outputs=[{"model": "dm-1026", "rewards": response.rewards[video_idx]}],
        plot_save_path=f"/data/sls/scratch/pschro/sole/video_frames_video_{video_idx}.mp4",
        video_frames_view1=video_frames_view1[video_idx],
        video_frames_view2=video_frames_view2[video_idx],
        video_frames_view3=video_frames_view3[video_idx],
        view_type=["left_wrist", "top", "right_wrist"],
        plot_width_relative_to_video=0.42, # works well when including 3 square views 
        fps=5,
    )



```

## Video frames with views stitched together

```python
import numpy as np

# stitching frames for each view from left to right: left_wrist, top, right_wrist 
video_frames = [
    [np.concatenate(frames, axis=1) for frames in zip(video1, video2, video3)]
    for video1, video2, video3 in zip(video_frames_view1, video_frames_view2, video_frames_view3,)
]


# video frames can be numpy arrays, torch tensors, or lists (with frames still as numpy arrays)
response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    video_frames=video_frames,
    verbose=True
)


# Plot rewards for each episode
for video_idx in range(len(video_frames_view1)):
    video_plot(
        outputs=[{"model": "dm-1026", "rewards": response.rewards[video_idx]}],
        plot_save_path=f"/data/sls/scratch/pschro/sole/video_frames_video_{video_idx}.mp4",
        video_frames_view1=video_frames_view1[video_idx],
        video_frames_view2=video_frames_view2[video_idx],
        video_frames_view3=video_frames_view3[video_idx],
        view_type=["left_wrist", "top", "right_wrist"],
        plot_width_relative_to_video=0.42, # works well when including 3 square views 
        fps=5,
    )



```


## Video paths with views already stitched
```python

# can use example_videos in this repo at https://github.com/Philip-MIT/sublinear/tree/main/example_videos/dream_machines

# VIDEO_DIR = '/path/to/example_videos/dream_machines'

video_paths=[
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_stitched.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000000_view_stitched.mp4",
    f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000025_view_stitched.mp4",
]


response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    video_paths=video_paths,
    verbose=True
)


for video_idx in range(len(video_paths)):
    video_plot(
        outputs=[{"model": "dm-1026", "rewards": response.rewards[video_idx]}],
        plot_save_path=f"/data/sls/scratch/pschro/sole/video_path_stitched_video_{video_idx}.mp4",
        video_path=video_paths[video_idx],
        plot_width_relative_to_video=0.42, # works well when including 3 square views 
        fps=5,
    )


```

