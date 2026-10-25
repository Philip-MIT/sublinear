# Sublinear Python SDK

The official Python client for submitting robot videos and trajectories to Sublinear foundation reward models.

## Installation

```bash
pip install -U --no-cache-dir sublinear
```

```python
from sublinear import Sublinear
```

## Create API key

Create an API key from the Sublinear dashboard (https://sublinear.dev/login).

## Package layout

This project uses a `src` layout:

```text
sublinear/
├── pyproject.toml
├── README.md
└── docs/
└── tests/
└── example_videos/
└── src/
    └── sublinear/
        ├── __init__.py
        └── core.py
```

