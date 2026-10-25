## Installation

The sublinear python package is actively being updated so please ensure you use the most updated version.

```bash
pip install -U --no-cache-dir sublinear
```

## Notes for Dream Machines
- Model is fine-tuned on the cropped 224x224 frames, with all 3 views always available
- Model performs best when predicting rewards at 30 timesteps per video (downsample_to=30)
- by default, client submits a maximum of 30 videos (max_frames=30) to the model
    - can try larger batch sizes, but may lead to crashing


## Compute and plot rewards for local lerobot dataset

Create an API key from the Sublinear dashboard (https://sublinear.dev/login), then submit a local lerobot dataset:

```python
from sublinear import Sublinear, video_plot
from lerobot.datasets import LeRobotDataset

# API_KEY="sk_...."

client = Sublinear(api_key=API_KEY)

DATA_DIR = '/path/to/data'
PLOT_OUTPUT_DIR = '/path/to/plots'

episode_ids = [0, 1, 2, 3]
dataset = LeRobotDataset(
    repo_id="local/actuator_unboxing_fullbox_eval_pi07_padmask_20261001", 
    root=f"{DATA_DIR}/DreamMachines/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_full"
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

# Print rewards for first episode in dataset
print(response.rewards[0])

# Plot rewards for each episode
for episode_id in episode_ids:
    video_plot(
        outputs=[{"model": "dm-1026", "rewards": response.rewards[episode_id]}],
        plot_save_path=f"{PLOT_OUTPUT_DIR}/video_plot_episode_{episode_id}.mp4",
        lerobot_dataset=dataset, 
        episode=episode_id,
        lerobot_view_map={
            "left_wrist": "observation.images.left_wrist",
            "top": "observation.images.top",
            "right_wrist": "observation.images.right_wrist",
        },
        plot_width_relative_to_video=0.42, # works well when including 3 square views 
        fps=5,
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
```

## Video paths separated across views

```python

# can use example_videos in this repo

VIDEO_DIR = '/path/to/example_videos/dream_machines'

video_path_view1=f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_left_wrist.mp4"
video_path_view2=f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_top.mp4"
video_path_view3=f"{VIDEO_DIR}/eval_actuator_unboxing_pi05_sweep_v2_01_freeze_01_episode_000073_view_right_wrist.mp4"


response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    video_paths_view1=[video_path_view1],
    video_paths_view2=[video_path_view2],
    video_paths_view3=[video_path_view3],
    view_type=["left_wrist", "top", "right_wrist"]
)

video_plot(
    outputs=[{"model": "dm-1026", "rewards": response.rewards[video_idx]}],
    plot_save_path=f"/data/sls/scratch/pschro/sole/video_plot_video_{video_idx}.mp4",
    video_path_view1=video_path_view1,
    video_path_view2=video_path_view2,
    video_path_view3=video_path_view3,
    view_type=["static", "left_wrist", "right_wrist"],
    plot_width_relative_to_video=0.42, # works well when including 3 square views 
    fps=5,
)

```

## Video frames separated across views


```python
from sublinear import load_video_frames

video_frames_view1=load_video_frames(video_path_view1, max_frames=float('inf'))
video_frames_view2=load_video_frames(video_path_view2, max_frames=float('inf'))
video_frames_view3=load_video_frames(video_path_view3, max_frames=float('inf'))

# video frames can be numpy arrays, torch tensors, or lists (with frames still as numpy arrays)

response = client.responses.create(
    model="dm-1026",
    task_description='Take an actuator from the box and place it in the tray',
    video_frames_view1=video_frames_view1,
    video_frames_view2=video_frames_view2,
    video_frames_view3=video_frames_view3,
    view_type=["left_wrist", "top", "right_wrist"]
)


# Print rewards for first episode in dataset
print(response.rewards[0])
len(response.rewards[0])
len(video_frames_view1)

# Plot rewards for each episode
for video_idx in range(len(video_frames_view1)):
    video_plot(
        outputs=[{"model": "dm-1026", "rewards": response.rewards[video_idx]}],
        plot_save_path=f"/data/sls/scratch/pschro/sole/video_plot_video_{video_idx}.mp4",
        video_path_view1=video_frames_view1[video_idx],
        video_path_view2=video_frames_view2[video_idx],
        video_path_view3=video_frames_view3[video_idx],
        view_type=["left_wrist", "top", "right_wrist"],
        plot_width_relative_to_video=0.42, # works well when including 3 square views 
        fps=5,
    )

```

