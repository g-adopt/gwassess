"""gwassess is a python package that implements analytical and reference solutions
to the Richards equation in 2D and 3D domains.
"""
__all__ = ['TracyRichardsSolution2D', 'TracyRichardsSolution3D',
           'VauclinRichardsSolution2D', 'CockettRichardsSolution3D']

from .tracy import TracyRichardsSolution2D, TracyRichardsSolution3D  # noqa: F401
from .vauclin import VauclinRichardsSolution2D  # noqa: F401
from .cockett import CockettRichardsSolution3D  # noqa: F401
