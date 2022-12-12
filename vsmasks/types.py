from __future__ import annotations

from typing import Protocol

from vstools import PlanesT, SingleOrArrOpt, vs, CustomEnum

__all__ = [
    'MorphoFunc',
    'XxpandMode'
]


class MorphoFunc(Protocol):
    def __call__(
        self, clip: vs.VideoNode, planes: PlanesT = ...,
        threshold: int | None = ..., coordinates: SingleOrArrOpt[int] = ...
    ) -> vs.VideoNode:
        ...


class XxpandMode(CustomEnum):
    """Expand/inpand mode"""

    RECTANGLE = object()
    """Rectangular shape"""

    ELLIPSE = object()
    """Elliptical shape"""

    LOSANGE = object()
    """Diamond shape"""
