# from .mayavi import MayaviBackend  # noqa F401
from .matplotlib import MatplotlibBackend  # noqa F401

__all__ = ["MatplotlibBackend"]

# TODO: Make it configurable
DefaultBackend = MatplotlibBackend
