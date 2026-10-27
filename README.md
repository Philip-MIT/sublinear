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

## Documentation

- [Complete `create()` and `video_plot()` guide](docs/create_and_video_plot.md)
- [Dream Machines three-view guide](docs/dream_machines.md)

## License

[MIT](LICENSE.md)

