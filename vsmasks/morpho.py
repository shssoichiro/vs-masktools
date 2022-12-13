from __future__ import annotations

from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Literal, Sequence

from vsexprtools import ExprOp, aka_expr_available, norm_expr
from vsrgtools.util import wmean_matrix
from vstools import (
    ConvMode, CustomIndexError, FuncExceptT, PlanesT, StrList, check_variable, copy_signature, core, fallback,
    inject_self, interleave_arr, iterate, vs
)

from .types import MorphoFunc, XxpandMode

__all__ = [
    'Morpho',
    'grow_mask'
]


def __minmax_method(  # type: ignore
    self: Morpho, src: vs.VideoNode, thr: int | float | None = None,
    coordinates: int | tuple[int, ConvMode] | Sequence[int] | None = [1] * 8,
    iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
    *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    ...


def __morpho_method(  # type: ignore
    self: Morpho, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: int | float | None = None,
    coordinates: int | tuple[int, ConvMode] | Sequence[int] = 5, multiply: float | None = None,
    *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    ...


def __morpho_method2(  # type: ignore
    self: Morpho, clip: vs.VideoNode, sw: int, sh: int | None = None, mode: XxpandMode = XxpandMode.RECTANGLE,
    thr: int | None = None, planes: PlanesT = None, *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    ...


@dataclass
class Morpho:
    planes: PlanesT = None
    func: FuncExceptT | None = None
    fast: bool | None = None

    def __post_init__(self) -> None:
        self._fast = fallback(self.fast, aka_expr_available) and aka_expr_available

    def _check_params(
        self, radius: int, thr: int | float | None, coordinates: int | tuple[int, ConvMode] | Sequence[int],
        planes: PlanesT, func: FuncExceptT
    ) -> tuple[FuncExceptT, PlanesT]:
        if radius < 1:
            raise CustomIndexError('radius has to be greater than 0!', func, radius)

        if isinstance(coordinates, (int, tuple)):
            size = coordinates if isinstance(coordinates, int) else coordinates[0]

            if size < 2:
                raise CustomIndexError('when int or tuple, coordinates has to be greater than 1!', func, coordinates)

            if not self._fast and size != 3:
                raise CustomIndexError(
                    'with fast=False or no akarin plugin, you must have coordinates=3!', func, coordinates
                )
        elif len(coordinates) != 8:
            raise CustomIndexError('when a list, coordinates must contain exactly 8 numbers!', func, coordinates)

        if thr is not None and thr < 0.0:
            raise CustomIndexError('thr must be a positive number!', func, coordinates)

        return self.func or func, self.planes if planes is None else planes

    @classmethod
    def _morpho_xx_imum(
        cls, thr: int | float | None, op: Literal[ExprOp.MIN, ExprOp.MAX],
        coordinates: int | tuple[int, ConvMode] | Sequence[int], multiply: float | None = None
    ) -> StrList:
        if isinstance(coordinates, (int, tuple)):
            if isinstance(coordinates, tuple):
                size, mode = coordinates
            else:
                size, mode = coordinates, ConvMode.SQUARE

            assert size > 1

            radius = size // 2

            exclude = list[tuple[int, int]]()

            if size % 2 == 0:
                exclude.extend((x, radius) for x in range(-radius, radius + 1))
                exclude.append((radius, radius - 1))
        else:
            coordinates = list(coordinates)
            radius, mode = 3, ConvMode.SQUARE

        matrix = ExprOp.matrix('x', radius, mode, exclude)

        if not isinstance(coordinates, (int, tuple)):
            matrix.insert(len(matrix) // 2, 1)
            matrix = StrList([x for x, coord in zip(matrix, coordinates) if coord])

        matrix = StrList(interleave_arr(matrix, op * matrix.mlength, 2))

        if thr is not None:
            matrix.append('x', thr, ExprOp.SUB, ExprOp.MAX)

        if multiply is not None:
            matrix.append(multiply, ExprOp.MUL)

        return matrix

    def _mm_func(
        self, src: vs.VideoNode, radius: int, planes: PlanesT, thr: int | float | None,
        coordinates: int | tuple[int, ConvMode] | Sequence[int], multiply: float | None,
        *, func: FuncExceptT, mm_func: MorphoFunc, op: Literal[ExprOp.MIN, ExprOp.MAX],
        **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coordinates, planes, func)

        if self._fast:
            mm_func = norm_expr  # type: ignore[assignment]
            kwargs.update(expr=self._morpho_xx_imum(thr, op, coordinates, multiply))
        elif isinstance(coordinates, (int, tuple)):
            if isinstance(coordinates, tuple):
                if coordinates[1] is not ConvMode.SQUARE:
                    raise CustomIndexError(
                        'with fast=False or no akarin plugin, you must have ConvMode.SQUARE!', func, coordinates
                    )

                coordinates = coordinates[0]

            if coordinates != 3:
                raise CustomIndexError(
                    'with fast=False or no akarin plugin, you must have coordinates=3!', func, coordinates
                )

            kwargs.update(coordinates=[1] * 8)

        if not self._fast and multiply is not None:
            orig_mm_func = mm_func

            @copy_signature(mm_func)
            def _mm_func(*args: Any, **kwargs: Any) -> Any:
                return orig_mm_func(*args, **kwargs).std.Expr(f'x {multiply} *')

            mm_func = _mm_func

        return iterate(src, mm_func, radius, planes=planes, **kwargs)

    @inject_self
    def xxpand_transform(
        self, clip: vs.VideoNode, op: Literal[ExprOp.MIN, ExprOp.MAX], sw: int, sh: int | None = None,
        mode: XxpandMode = XxpandMode.RECTANGLE, thr: int | None = None,
        planes: PlanesT = None, *, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        func, planes = self._check_params(1, thr, 3, planes, func or self.xxpand_transform)

        sh = fallback(sh, sw)

        if op not in {ExprOp.MIN, ExprOp.MAX}:
            raise NotImplementedError

        function = self.maximum if op is ExprOp.MAX else self.minimum

        for (wi, hi) in zip_longest(range(sw, -1, -1), range(sh, -1, -1), fillvalue=0):
            if wi > 0 and hi > 0:
                if mode == XxpandMode.LOSANGE or (mode == XxpandMode.ELLIPSE and wi % 3 != 1):
                    coordinates = [0, 1, 0, 1, 1, 0, 1, 0]
                else:
                    coordinates = [1] * 8
            elif wi > 0:
                coordinates = [0, 0, 0, 1, 1, 0, 0, 0]
            elif hi > 0:
                coordinates = [0, 1, 0, 0, 0, 0, 1, 0]
            else:
                break

            clip = function(clip, thr, coordinates, planes=planes, func=func)

        return clip

    @inject_self
    @copy_signature(__minmax_method)
    def maximum(
        self, src: vs.VideoNode, thr: int | float | None = None,
        coordinates: int | tuple[int, ConvMode] | Sequence[int] | None = None,
        iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.dilation(src, iterations, planes, thr, coordinates or ([1] * 8), multiply, func=func, **kwargs)

    @inject_self
    @copy_signature(__minmax_method)
    def minimum(
        self, src: vs.VideoNode, thr: int | float | None = None,
        coordinates: int | tuple[int, ConvMode] | Sequence[int] | None = None,
        iterations: int = 1, multiply: float | None = None, planes: PlanesT = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.erosion(src, iterations, planes, thr, coordinates or ([1] * 8), multiply, func=func, **kwargs)

    @inject_self
    @copy_signature(__morpho_method)
    def dilation(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self._mm_func(*args, func=func or self.dilation, mm_func=core.std.Maximum, op=ExprOp.MAX, **kwargs)

    @inject_self
    @copy_signature(__morpho_method)
    def erosion(self, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self._mm_func(*args, func=func or self.erosion, mm_func=core.std.Minimum, op=ExprOp.MIN, **kwargs)

    @inject_self
    @copy_signature(__morpho_method2)
    def expand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self.xxpand_transform(clip, ExprOp.MIN, *args, func=func, **kwargs)

    @inject_self
    @copy_signature(__morpho_method2)
    def inpand(self, clip: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        return self.xxpand_transform(clip, ExprOp.MAX, *args, func=func, **kwargs)

    @inject_self
    @copy_signature(__morpho_method)
    def closing(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        func = func or self.closing

        dilated = self.dilation(src, *args, func=func, **kwargs)
        eroded = self.erosion(dilated, *args, func=func, **kwargs)

        return eroded

    @inject_self
    @copy_signature(__morpho_method)
    def opening(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        func = func or self.closing

        eroded = self.erosion(src, *args, func=func, **kwargs)
        dilated = self.dilation(eroded, *args, func=func, **kwargs)

        return dilated

    @inject_self
    def gradient(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: int | float | None = None,
        coordinates: int | tuple[int, ConvMode] | Sequence[int] = 5, multiply: float | None = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coordinates, planes, func or self.gradient)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{dilated} {eroded} -', planes,
                dilated=self._morpho_xx_imum(thr, ExprOp.MAX, coordinates),
                eroded=self._morpho_xx_imum(thr, ExprOp.MIN, coordinates)
            )

        eroded = self.erosion(src, radius, planes, thr, coordinates, multiply, func=func, **kwargs)
        dilated = self.dilation(src, radius, planes, thr, coordinates, multiply, func=func, **kwargs)

        return norm_expr([dilated, eroded], 'x y -', planes)

    @inject_self
    @copy_signature(__morpho_method)
    def top_hat(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        opened = self.opening(src, *args, func=func or self.top_hat, **kwargs)

        return norm_expr([src, opened], 'x y -', kwargs.get('planes', args[1] if len(args) > 1 else None))

    @inject_self
    @copy_signature(__morpho_method)
    def black_hat(self, src: vs.VideoNode, *args: Any, func: FuncExceptT | None = None, **kwargs: Any) -> vs.VideoNode:
        closed = self.closing(src, *args, func=func or self.black_hat, **kwargs)

        return norm_expr([closed, src], 'x y -', kwargs.get('planes', args[1] if len(args) > 1 else None))

    @inject_self
    def outer_hat(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: int | float | None = None,
        coordinates: int | tuple[int, ConvMode] | Sequence[int] = 5, multiply: float | None = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coordinates, planes, func or self.outer_hat)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{dilated} x -', planes,
                dilated=self._morpho_xx_imum(thr, ExprOp.MAX, coordinates)
            )

        dilated = self.dilation(src, radius, planes, thr, coordinates, multiply, func=func, **kwargs)

        return norm_expr([dilated, src], 'x y -', planes)

    @inject_self
    def inner_hat(
        self, src: vs.VideoNode, radius: int = 1, planes: PlanesT = None, thr: int | float | None = None,
        coordinates: int | tuple[int, ConvMode] | Sequence[int] = 5, multiply: float | None = None,
        *, func: FuncExceptT | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        func, planes = self._check_params(radius, thr, coordinates, planes, func or self.inner_hat)

        if radius == 1 and self._fast:
            return norm_expr(
                src, '{eroded} x -', planes,
                eroded=self._morpho_xx_imum(thr, ExprOp.MIN, coordinates)
            )

        eroded = self.erosion(src, radius, planes, thr, coordinates, multiply, func=func, **kwargs)

        return norm_expr([eroded, src], 'x y -', planes)


def grow_mask(
    mask: vs.VideoNode, radius: int = 1, multiply: float = 1.0,
    planes: PlanesT = None, coordinates: int | tuple[int, ConvMode] | Sequence[int] = 5,
    thr: int | float | None = None, *, func: FuncExceptT | None = None, **kwargs: Any
) -> vs.VideoNode:
    func = func or grow_mask

    assert check_variable(mask, func)

    morpho = Morpho(planes)

    kwargs.update(thr=thr, coordinates=coordinates, func=func)

    closed = morpho.closing(mask, **kwargs)
    dilated = morpho.dilation(closed, **kwargs)
    outer = morpho.outer_hat(dilated, radius, **kwargs)

    blurred = outer.std.Convolution(wmean_matrix, planes=planes)

    if multiply != 1.0:
        return blurred.std.Expr(f'x {multiply} *')

    return blurred
