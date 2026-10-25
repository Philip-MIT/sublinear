from .core import *

__all__ = []

try:
    from importlib.metadata import version
    __version__ = version("sublinear")
except Exception:
    __version__ = "0.1.4"

