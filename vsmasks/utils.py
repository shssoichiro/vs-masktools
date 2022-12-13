from __future__ import annotations

from vsexprtools import aka_expr_available, norm_expr
from vsrgtools import box_blur, gauss_blur
from vstools import (
    CustomValueError, FrameRangeN, FrameRangesN, FuncExceptT, check_variable, get_peak_value, insert_clip,
    replace_ranges, vs
)

__all__ = [
    'squaremask',
    'replace_squaremask',
    'freeze_replace_squaremask',
]

def squaremask(
    clip: vs.VideoNode, width: int, height: int, offset_x: int, offset_y: int, invert: bool = False,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or squaremask

    assert check_variable(clip, func)

    mask_format = clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)

    if offset_x + width > clip.width or offset_y + height > clip.height:
        raise CustomValueError('mask exceeds clip size!')

    if aka_expr_available:
        base_clip = clip.std.BlankClip(mask_format.id, 1, color=0, keep=True)

        invert_str = 'range_max x' if invert else 'x range_max'

        right = clip.width - (offset_x + 1)
        bottom = clip.height - (offset_y + 1)

        mask = norm_expr(
            base_clip, f'X {offset_x} < X {right} > or Y {offset_y} < Y {bottom} > or or {invert_str} ?',
            force_akarin=func
        )
    else:
        base_clip = clip.std.BlankClip(
            width, height, mask_format.id, 1, color=get_peak_value(clip), keep=True
        )
        mask = base_clip.std.AddBorders(
            offset_x, clip.width - width - offset_x, offset_y, clip.height - height - offset_y
        )
        if invert:
            mask = mask.std.Invert()

    if clip.num_frames == 1:
        return mask

    return mask.std.Loop(clip.num_frames)


def replace_squaremask(
    clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: tuple[int, int, int, int],
    ranges: FrameRangeN | FrameRangesN | None = None, blur_sigma: int | float | None = None,
    invert: bool = False, func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or replace_squaremask

    assert check_variable(clipa, func) and check_variable(clipb, func)

    mask = squaremask(clipb[0], *mask_params, invert, func)

    if isinstance(blur_sigma, int):
        mask = box_blur(mask, blur_sigma)
    elif isinstance(blur_sigma, float):
        mask = gauss_blur(mask, blur_sigma)

    merge = clipa.std.MaskedMerge(clipb, mask.std.Loop(clipa.num_frames))

    return replace_ranges(clipa, merge, ranges)


def freeze_replace_squaremask(
    mask: vs.VideoNode, insert: vs.VideoNode, mask_params: tuple[int, int, int, int],
    frame: int, frame_range: tuple[int, int]
) -> vs.VideoNode:
    start, end = frame_range

    masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)

    return insert_clip(mask, masked_insert * (end - start + 1), start)
