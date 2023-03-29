from __future__ import annotations

from vsrgtools import gauss_blur, box_blur
from vstools import vs, get_y, scale_value, depth, DitherType, ColorRange


def flat_mask(src: vs.VideoNode, radius: int = 5, thr: int = 3, gauss: bool = False) -> vs.VideoNode:
    luma = get_y(src)

    blur = gauss_blur(luma, radius * 0.361083333) if gauss else box_blur(luma, radius)

    mask = depth(luma, 8).abrz.AdaptiveBinarize(depth(blur, 8), scale_value(thr, 32, 8))

    return depth(mask, luma, dither_type=DitherType.NONE, range_in=ColorRange.FULL, range=ColorRange.FULL)
