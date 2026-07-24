#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/24 11:21 
# @File    : _grd_extract_wrapper.py

from __future__ import annotations

import datetime as dt
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

from sagea.core._namespace_accessor import (
    BaseNamespaceAccessor,
    NamespaceAccessorDescriptor,
    namespace_method,
    NamespaceMethodKind,
)
from sagea.utils import MathTool

if TYPE_CHECKING:
    from .grid import GRD


def extract_method(
        *,
        summary: str | None = None,
        order: int = 0,
        kind: NamespaceMethodKind = "instance",
        show: bool = True,
) -> Callable:
    """
    Mark a method as a public GRD extract method.

    Parameters
    ----------
    summary:
        Short description shown in GRD.extract.help() / grd.extract.help().
        If None, the first line of method docstring will be used.
    order:
        Display order in help text.
    kind:
        Usually "class" for extract methods.
    show:
        Whether shown in help().
    """
    return namespace_method(
        namespace="extract",
        summary=summary,
        order=order,
        kind=kind,
        show=show,
    )


class _GRDExtractAccessor(BaseNamespaceAccessor):
    """
    Accessor namespace for GRD extract methods.

    Recommended usage
    -----------------
    GRD.extract.maskGRD(...)

    Help
    ----
    print(GRD.extract.help())
    print(grd.extract.help())

    Notes
    -----
    Most extract methods are instance methods.
    Calling them from a class is intentionally disallowed.
    """

    _namespace_name = "extract"

    @property
    def _grd_cls(self) -> type["GRD"]:
        return self._owner

    def __call__(self, *args, **kwargs):
        raise TypeError(
            f"`{self._class_usage_name}` is a extract-method namespace, "
            "not a standalone constructor. "
            f"Use `{self._class_usage_name}.<method_name>(...)` instead."
        )

    # ============================================================
    # Class generator methods
    # ============================================================

    @extract_method(
        kind="instance",
        summary="Extract GRD signals in given mask: GRD; "
                "return a 2-d array with shape (length of mask, length of self)",
        order=10,
    )
    def maskGRD(self, mask: GRD,
                average: bool = True,
                ) -> np.ndarray:
        """
        Extract GRD signals in given mask: GRD
        """
        grd = self._obj

        mask_value = mask.value

        if not (np.allclose(mask.lat, grd.lat) and np.allclose(mask.lon, grd.lon)):
            raise ValueError("mask must have same lat and lon with self.lat and self.lon")

        result = np.array([
            MathTool.global_integral(grd.value * mask_value[i], grd.lat, grd.lon)
            for i in range(len(mask_value))
        ])

        if average:
            for i in range(mask_value.shape[0]):
                result[i, :] = result[i, :] / MathTool.get_acreage(mask_value[i])

        return result


class GRDExtractAccessorDescriptor(NamespaceAccessorDescriptor[_GRDExtractAccessor]):
    """
    Descriptor for GRD.extractor and grd.extractor.
    """

    def __init__(self) -> None:
        super().__init__(_GRDExtractAccessor)
