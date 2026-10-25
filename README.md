# Sublinear Python SDK

The official Python client for submitting robot videos and trajectories to Sublinear foundation reward models.

## Installation

```bash
pip install sublinear
```

The PyPI distribution is named `sublinear`, while the Python import package is named `sublinear`:

```python
from sublinear import Sublinear
```

## Quick start

Create an API key from the Sublinear dashboard, then submit a local MP4 video:

```python
import os
from sublinear import Sublinear
client = Sublinear(api_key=os.environ["SUBLINEAR_API_KEY"])
response = client.responses.create(model="sole-r1", task_description="Fold the clothing.", video_path="episode.mp4")
print(response.rewards)
print(response.success)
```

You can also provide RGB video frames directly as a list or NumPy array with shape `(frames, height, width, 3)`:

```python
import os
import numpy as np
from sublinear import Sublinear
client = Sublinear(api_key=os.environ["SUBLINEAR_API_KEY"])
video_frames = np.load("video_frames.npy")
response = client.responses.create(model="robometer", task_description="Fold the clothing.", video_frames=video_frames)
print(response.rewards)
print(response.success)
```

## Multiple videos

Pass `video_paths` to process multiple videos. The SDK submits jobs concurrently, up to `max_parallel_jobs` at a time.

```python
import os
from sublinear import Sublinear
client = Sublinear(api_key=os.environ["SUBLINEAR_API_KEY"], max_parallel_jobs=8)
response = client.responses.create(model="sole-r1", task_description="Pick up the cube.", video_paths=["episode_1.mp4", "episode_2.mp4"])
print(response.rewards)
print(response.success)
```

## Available models

- `sole-r1` — reasoning-based robot trajectory evaluation
- `robometer` — dense task-progress reward estimation
- `topreward` — visual reward estimation across robot tasks
- `roboreward` — general-purpose robot reward prediction

## Documentation

Full documentation and additional input formats are available at [sublinear.dev/docs](https://sublinear.dev/docs).

## Package layout

This project uses a `src` layout:

```text
sublinear/
├── pyproject.toml
├── README.md
└── src/
    └── sublinear/
        ├── __init__.py
        └── core.py
```

Your `src/sublinear/__init__.py` should expose the public client:

```python
from .core import Sublinear
__all__ = ["Sublinear"]
```

## Development

Install the package locally in editable mode:

```bash
python -m pip install -e .
```

Build the source distribution and wheel:

```bash
python -m pip install --upgrade build
python -m build
```

Before publishing, verify the package metadata and archives:

```bash
python -m pip install --upgrade twine
python -m twine check dist/*
```

Publish to PyPI:

```bash
python -m twine upload dist/*
```

## License

MIT

