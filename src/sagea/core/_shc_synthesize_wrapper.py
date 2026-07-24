#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/22 12:24
# @File    : _shc_synthesize_wrapper.py

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import numpy as np

from sagea.harmonics.harmonic import HarmonicDiscrete

from sagea.harmonics.transform import shc_to_grid
from sagea.core.grid import GRD

from sagea.core._namespace_accessor import (
    BaseNamespaceAccessor,
    NamespaceAccessorDescriptor,
    NamespaceMethodKind,
    namespace_method,
)

if TYPE_CHECKING:
    from .shc import SHC

NAMESPACE = "synthesize"


def synthesize_method(
        *,
        summary: str | None = None,
        order: int = 0,
        kind: NamespaceMethodKind = "instance",
        show: bool = True,
) -> Callable:
    return namespace_method(
        namespace=NAMESPACE,
        summary=summary,
        order=order,
        kind=kind,
        show=show,
    )


class _SHCSynthesizeAccessor(BaseNamespaceAccessor):
    _namespace_name = NAMESPACE

    @property
    def _shc(self) -> "SHC":
        if self._obj is None:
            raise RuntimeError(
                f"This {NAMESPACE} method requires an SHC instance."
            )
        return self._obj

    @synthesize_method(
        kind="instance",
        summary="harmonic synthesis the spherical coefficients to class GRD; return a new GRD class",
    )
    def to_grid(self, grid_space, grid_type=None) -> "GRD":
        """
        SHC -> Grid.
        """

        shc = self._shc
        cqlm, sqlm = shc.cs2d

        grid_value, lat, lon = shc_to_grid(
            cqlm=cqlm, sqlm=sqlm,
            lmax=shc.lmax,
            grid_space=grid_space,
            grid_type=grid_type,
        )

        return GRD(grid_value, lat, lon)

    @synthesize_method(
        kind="instance",
        summary="harmonic-synthesis evaluate at discrete points",
    )
    def evaluate(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        return: 1d-array with the same shape as lat and lon.
        ----------
        input:
        lat: latitudes as 1d-array, in unit [degree]
        lon: longitudes as 1d-array, in unit [degree]
        ----------
        note:
        lat and lon are assumed to be in degrees and in the same length.
        """
        assert lat.shape == lon.shape
        assert lat.ndim == 1

        shc = self._shc
        cqlm, sqlm = shc.cs2d

        har = HarmonicDiscrete(
            lmax=shc.lmax,
            lat=lat,
            lon=lon,
        )

        results = har.synthesis(cqlm, sqlm)

        if results.ndim == 1:
            results = results[None, :]

        return results


class SHCSynthesizeAccessorDescriptor(
    NamespaceAccessorDescriptor[_SHCSynthesizeAccessor]
):
    def __init__(self) -> None:
        super().__init__(_SHCSynthesizeAccessor)
