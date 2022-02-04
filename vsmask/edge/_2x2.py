__all__ = [
    'Matrix2x2',
    'Roberts'
]

from abc import ABC

from ._abstract import EdgeDetect, EuclidianDistanceMatrixDetect


class Matrix2x2(EdgeDetect, ABC):
    ...


class Roberts(EuclidianDistanceMatrixDetect, Matrix2x2):
    """Lawrence Roberts operator. 2x2 matrices computed in 3x3 matrices."""
    matrices = [
        [0, 0, 0, 0, 1, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 1, 0, -1, 0]
    ]
