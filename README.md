# Sublinear Python SDK

Python client for computing per-frame robot-task rewards with Sublinear foundation reward models.

## Install

```bash
pip install -U sublinear
```

For LeRobot datasets:

```bash
pip install -U "sublinear[lerobot]"
```

Requires Python 3.10 or newer.

## API key

Create an API key at [sublinear.dev/login](https://sublinear.dev/login), then store it in an environment variable:

```bash
export SUBLINEAR_API_KEY="sk_..."
```

## Quick start

```python
import os

from sublinear import Sublinear

client = Sublinear(
    api_key=os.environ["SUBLINEAR_API_KEY"]
)

response = client.responses.create(
    model="dm-1026",
    task_description=(
        "Take an actuator from the box and place it in the tray"
    ),
    video_path="/path/to/episode.mp4",
    max_frames=50,
    downsample_to=30,
)

rewards = response.rewards[0]
print(f"{len(rewards)=}")
```

`create()` is synchronous: it packages and uploads the input, waits for inference, and returns full-length per-frame rewards.

Supported inputs include:

- One or more local video paths.
- NumPy or Torch frame arrays.
- One, two, or three synchronized camera views.
- Local or downloaded, video-backed LeRobot datasets.
- Videos whose views are already stitched horizontally.

Use exactly one input family per request.

## Documentation

- [Complete `create()` and `video_plot()` guide](docs/create_and_video_plot.md)
- [Dream Machines three-view guide](docs/dream_machines.md)

## License

[MIT](LICENSE.md)

