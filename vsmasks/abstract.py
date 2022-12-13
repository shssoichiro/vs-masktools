from abc import ABC, abstractmethod

from vstools import (
    ColorRange, FrameRangeN, FrameRangesN, Position, Size, check_variable, core, depth, replace_ranges, vs
)

from .utils import squaremask

__all__ = [
    'Mask',
    'BoundingBox',
]


class Mask(ABC):
    ...


class BoundingBox(Mask):
    pos: Position
    size: Size
    invert: bool

    def __init__(self, pos: tuple[int, int] | Position, size: tuple[int, int] | Size, invert: bool = False) -> None:
        self.pos, self.size, self.invert = Position(pos), Size(size), invert

    def get_mask(self, ref: vs.VideoNode) -> vs.VideoNode:
        return squaremask(ref, self.size.x, self.size.y, self.pos.x, self.pos.y, self.invert, self.get_mask)


