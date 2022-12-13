from __future__ import annotations

from functools import partial

from vsexprtools import norm_expr
from vsrgtools.util import wmean_matrix
from vstools import check_variable, core, depth, get_depth, get_peak_value, iterate, plane, scale_thresh, vs

from .edge import EdgeDetect, Prewitt

__all__ = [
    'ringing_mask'
]


def ringing_mask(
    clip: vs.VideoNode,
    rad: int = 2, brz: float = 0.35,
    thmi: float = 0.315, thma: float = 0.5,
    thlimi: float = 0.195, thlima: float = 0.392,
    credit_mask: vs.VideoNode | EdgeDetect = Prewitt()
) -> vs.VideoNode:
    assert check_variable(clip, ringing_mask)

    smax = get_peak_value(clip)

    thmi, thma, thlimi, thlima = (
        scale_thresh(t, clip) for t in [thmi, thma, thlimi, thlima]
    )

    if isinstance(credit_mask, vs.VideoNode):
        edgemask = depth(credit_mask, get_depth(clip))  # type: ignore
    elif isinstance(credit_mask, EdgeDetect):
        edgemask = credit_mask.edgemask(plane(clip, 0))

    edgemask = plane(edgemask, 0).std.Limiter()

    light = norm_expr(edgemask, f'x {thlimi} - {thma - thmi} / {smax} *')
    shrink = iterate(light, core.std.Maximum, rad)
    shrink = shrink.std.Binarize(scale_thresh(brz, clip))
    shrink = iterate(shrink, core.std.Minimum, rad)
    shrink = iterate(shrink, partial(core.std.Convolution, matrix=wmean_matrix), 2)

    strong = norm_expr(edgemask, f'x {thmi} - {thlima - thlimi} / {smax} *')
    expand = iterate(strong, core.std.Maximum, rad)

    mask = norm_expr([expand, strong, shrink], 'x y z max - 2 *')
    mask = mask.std.Convolution(wmean_matrix)

    return norm_expr(mask, 'x 2 *').std.Limiter()
