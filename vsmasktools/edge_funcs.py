from __future__ import annotations

from functools import partial
from typing import Any, Sequence

from vsexprtools import ExprOp, ExprToken, norm_expr
from vsrgtools import gauss_blur
from vsrgtools.util import wmean_matrix
from vstools import check_variable, core, get_peak_value, get_y, iterate, plane, scale_value, vs

from .details import multi_detail_mask
from .edge import FDoGTCanny, Kirsch, Prewitt
from .spat_funcs import retinex
from .morpho import Morpho
from .types import Coordinates, GenericMaskT
from .utils import normalize_mask

__all__ = [
    'ringing_mask',

    'luma_mask', 'luma_credit_mask',

    'tcanny_retinex',

    'limited_linemask'
]


def ringing_mask(
    clip: vs.VideoNode,
    rad: int = 2, brz: float = 0.35,
    thmi: float = 0.315, thma: float = 0.5,
    thlimi: float = 0.195, thlima: float = 0.392,
    credit_mask: GenericMaskT = Prewitt, **kwargs: Any
) -> vs.VideoNode:
    assert check_variable(clip, ringing_mask)

    thmi, thma, thlimi, thlima = (
        scale_value(t, 32, clip) for t in [thmi, thma, thlimi, thlima]
    )

    edgemask = normalize_mask(credit_mask, plane(clip, 0), **kwargs).std.Limiter()

    light = norm_expr(edgemask, f'x {thlimi} - {thma - thmi} / {ExprToken.RangeMax} *')

    shrink = Morpho.dilation(light, rad)
    shrink = Morpho.binarize(shrink, brz)
    shrink = Morpho.erosion(shrink, 2)
    shrink = iterate(shrink, partial(core.std.Convolution, matrix=wmean_matrix), 2)

    strong = norm_expr(edgemask, f'x {thmi} - {thlima - thlimi} / {ExprToken.RangeMax} *')
    expand = Morpho.dilation(strong, rad)

    mask = norm_expr([expand, strong, shrink], 'x y z max -')

    return ExprOp.convolution('x', wmean_matrix, premultiply=2, multiply=2, clamp=True)(mask)


def luma_mask(clip: vs.VideoNode, thr_lo: float, thr_hi: float, invert: bool = True) -> vs.VideoNode:
    peak = get_peak_value(clip)

    lo, hi = (peak, 0) if invert else (0, peak)
    inv_pre, inv_post = (peak, '-') if invert else ('', '')

    thr_lo, thr_hi = scale_value(thr_lo, 32, clip), scale_value(thr_hi, 32, clip)

    return norm_expr(
        get_y(clip),
        f'x {thr_lo} < {lo} x {thr_hi} > {hi} {inv_pre} x {thr_lo} - {thr_lo} {thr_hi} - / {peak} * {inv_post} ? ?'
    )


def luma_credit_mask(
    clip: vs.VideoNode, thr: float = 0.9, edgemask: GenericMaskT = FDoGTCanny, draft: bool = False, **kwargs: Any
) -> vs.VideoNode:
    y = plane(clip, 0)

    edge_mask = normalize_mask(edgemask, y, **kwargs)

    credit_mask = norm_expr([edge_mask, y], f'y {scale_value(thr, 32, y)} > y 0 ? x min')

    if not draft:
        credit_mask = Morpho.maximum(credit_mask, iterations=4)
        credit_mask = Morpho.inflate(credit_mask, iterations=2)

    return credit_mask


def tcanny_retinex(
    clip: vs.VideoNode, thr: float, sigma: Sequence[float] = [50, 200, 350], blur_sigma: float = 1.0
) -> vs.VideoNode:
    blur = gauss_blur(clip, blur_sigma)

    msrcp = retinex(blur, sigma, upper_thr=thr, fast=True, func=tcanny_retinex)

    tcunnied = msrcp.tcanny.TCanny(mode=1, sigma=1)

    return Morpho.minimum(tcunnied, None, Coordinates.CORNERS)


def limited_linemask(
    clip: vs.VideoNode,
    sigmas: list[float] = [0.000125, 0.0025, 0.0055],
    detail_sigmas: list[float] = [0.011, 0.013],
    edgemasks: Sequence[GenericMaskT] = [Kirsch],
    **kwargs: Any
) -> vs.VideoNode:
    clip_y = plane(clip, 0)

    return ExprOp.ADD(
        (normalize_mask(edge, clip_y, **kwargs) for edge in edgemasks),
        (tcanny_retinex(clip_y, s) for s in sigmas),
        (multi_detail_mask(clip_y, s) for s in detail_sigmas)
    )
