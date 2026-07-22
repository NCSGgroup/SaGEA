#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/22 11:44 
# @File    : _shc_correction_wrapper.py

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from sagea.utils import MathTool

from sagea.core._namespace_accessor import (
    BaseNamespaceAccessor,
    NamespaceAccessorDescriptor,
    NamespaceMethodKind,
    namespace_method,
)

if TYPE_CHECKING:
    from .shc import SHC

NAMESPACE = "correction"


def correction_method(
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


class _SHCCorrectionAccessor(BaseNamespaceAccessor):
    _namespace_name = NAMESPACE

    @property
    def _shc(self) -> "SHC":
        if self._obj is None:
            raise RuntimeError(
                f"This {NAMESPACE} method requires an SHC instance."
            )
        return self._obj

    # ============================================================
    # Instance/class methods templates
    # ============================================================
    # @correction_method(
    #     kind="instance",
    #     summary="Example instance method.",
    #     order=100,
    # )
    # def some_instance_method(self, *, inplace: bool = False) -> "SHC":
    #     shc = self._shc
    #     obj = shc if inplace else shc.copy()
    #
    #     # do something
    #     return obj
    #
    # @correction_method(
    #     kind="class",
    #     summary="Example class method.",
    #     order=200,
    # )
    # def some_class_method(self):
    #     cls = self._owner
    #
    #     # do something with SHC class
    #     return cls
    @correction_method(
        kind="instance",
        summary="Geometric correction, see Yang et al., 2022, "
                "https://doi.org/10.1007/s00190-022-01683-0",
    )
    def geometric(
            self,
            assumption="ActualEarth",
            field_type="EWH",
            love_number=None,
            love_number_method="Wang",
            grid_space: float | None = 0.5,
            grid_type=None,
            orography=None,
            undulation=None,

            phisfc_file: str | None = None,
            gif48_file: str | None = None,
            geoid_lmax: int = 160,
            auto_load_actual_earth: bool = False,

            iter_max: int = 4,
            vmax: float = 2.5,
            vmin: float = 0.0,
            inplace: bool = False,
            verbose: bool = False,
    ):
        from sagea.corrections.geometric_correction.geometric import (
            apply_geometric_correction,
        )

        shc = self._shc
        cls = self._owner

        obj = shc if inplace else shc.copy()

        if love_number is None:
            love_number = cls._load_love_number(
                lmax=obj.lmax,
                love_number_method=love_number_method,
            )

        cqlm, sqlm = obj.cs2d

        cqlm_corr, sqlm_corr = apply_geometric_correction(
            cqlm=cqlm,
            sqlm=sqlm,
            love_number=love_number,
            assumption=assumption,
            field_type=field_type,
            love_number_method=love_number_method,
            grid_space=grid_space,
            grid_type=grid_type,
            orography=orography,
            undulation=undulation,

            phisfc_file=phisfc_file,
            gif48_file=gif48_file,
            geoid_lmax=geoid_lmax,
            auto_load_actual_earth=auto_load_actual_earth,

            iter_max=iter_max,
            vmax=vmax,
            vmin=vmin,
            verbose=verbose,
        )

        obj._values = MathTool.cs_combine_to_triangle_1d(cqlm_corr, sqlm_corr)

        return obj


class SHCCorrectionAccessorDescriptor(
    NamespaceAccessorDescriptor[_SHCCorrectionAccessor]
):
    def __init__(self) -> None:
        super().__init__(_SHCCorrectionAccessor)
