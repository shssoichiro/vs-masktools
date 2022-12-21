from __future__ import annotations

from typing import Any

from vsexprtools import ExprOp
from vskernels import Bicubic, Catrom, Kernel, KernelT
from vsrgtools import RemoveGrainMode, bilateral, removegrain
from vstools import (
    CustomValueError, FuncExceptT, KwargsT, check_variable, depth, expect_bits, get_w, get_y, insert_clip, iterate,
    scale_value, vs
)

from .edge import ExLaplacian4
from .morpho import Morpho
from .types import XxpandMode


def diff_rescale(
    clip: vs.VideoNode, height: int, kernel: KernelT = Catrom,
    thr: int | float = 55, expand: int = 2, func: FuncExceptT | None = None
) -> vs.VideoNode:
    func = func or diff_rescale

    assert check_variable(clip, func)

    kernel = Kernel.ensure_obj(kernel, func)

    pre, bits = expect_bits(get_y(clip), 32)

    thr = scale_value(thr, bits, 32, scale_offsets=True)

    descale = kernel.descale(pre, get_w(height), height)
    rescale = kernel.scale(descale, clip.width, clip.height)

    diff = ExprOp.mae(clip)(pre, rescale)

    mask = iterate(diff, removegrain, 2, RemoveGrainMode.MINMAX_AROUND2)
    mask = mask.std.Expr(f'x 2 4 pow * {thr} < 0 1 ?')

    mask = Morpho.expand(mask, 2 + expand, mode=XxpandMode.ELLIPSE).std.Deflate()

    return depth(mask, bits)


def diff_creditless(
    src_clip: vs.VideoNode, credit_clip: vs.VideoNode, nc_clip: vs.VideoNode,
    start_frame: int, thr: int, expand: int = 2, *, prefilter: bool | int = False,
    func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or diff_creditless

    assert check_variable(src_clip, func)

    clips = [credit_clip, nc_clip]

    if prefilter:
        sigma = 5 if prefilter is True else prefilter
        kwargs |= KwargsT(sigmaS=((sigma ** 2 - 1) / 12) ** 0.5, sigmaR=sigma / 10) | kwargs
        clips = [bilateral(c, **kwargs) for c in clips]

    dst_fmt = src_clip.format.replace(subsampling_w=0, subsampling_h=0)
    diff_fmt = dst_fmt.replace(color_family=vs.GRAY)

    diff = ExprOp.mae(dst_fmt)(
        (Bicubic.resample(c, dst_fmt) for c in clips),
        format=diff_fmt, split_planes=True
    )

    mask = ExLaplacian4().edgemask(diff).std.Binarize(thr)
    mask = Morpho.expand(mask, 2 + expand, mode=XxpandMode.ELLIPSE)

    if src_clip.num_frames == mask.num_frames:
        return mask

    blank = src_clip.std.BlankClip(format=diff_fmt.id, keep=True)

    return insert_clip(blank, mask, start_frame)


def diff_creditless_oped(
    ep: vs.VideoNode, ncop: vs.VideoNode, nced: vs.VideoNode,
    opstart: int | None = None, opend: int | None = None,
    edstart: int | None = None, edend: int | None = None,
    func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or diff_creditless_oped

    op_mask = ed_mask = None

    kwargs |= KwargsT(thr=25, expand=4, prefilter=False, func=func) | kwargs

    if opstart is not None and opend is not None:
        op_mask = diff_creditless(ep, ep[opstart:opend + 1], ncop[:opend - opstart + 1], opstart, **kwargs)

    if edstart is not None and edend is not None:
        ed_mask = diff_creditless(ep, ep[edstart:edend + 1], nced[:edend - edstart + 1], edstart, **kwargs)

    if op_mask and ed_mask:
        return ExprOp.ADD.combine(op_mask, ed_mask)
    elif op_mask or ed_mask:
        return op_mask or ed_mask  # type: ignore

    raise CustomValueError(
        'You must specify one or both of ("opstart", "opend"), ("edstart", "edend")', func
    )
