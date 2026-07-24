#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/20 15:31 
# @File    : _shc_generator_wrapper.py

#!/usr/bin/env python
# coding=utf-8

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

if TYPE_CHECKING:
    from .shc import SHC


def generator_method(
    *,
    summary: str | None = None,
    order: int = 0,
    kind: NamespaceMethodKind = "class",
    show: bool = True,
) -> Callable:
    """
    Mark a method as a public SHC generator method.

    Parameters
    ----------
    summary:
        Short description shown in SHC.generate.help() / shc.generate.help().
        If None, the first line of method docstring will be used.
    order:
        Display order in help text.
    kind:
        Usually "class" for generator methods.
    show:
        Whether shown in help().
    """
    return namespace_method(
        namespace="generate",
        summary=summary,
        order=order,
        kind=kind,
        show=show,
    )


class _SHCGeneratorAccessor(BaseNamespaceAccessor):
    """
    Accessor namespace for SHC generator methods.

    Recommended usage
    -----------------
    SHC.generate.from_array(...)
    SHC.generate.from_trend(...)
    SHC.generate.normal_from_vcm(...)

    Help
    ----
    print(SHC.generate.help())
    print(shc.generate.help())

    Notes
    -----
    Most generator methods are class methods.
    Calling them from an instance is intentionally disallowed.
    """

    _namespace_name = "generate"

    @property
    def _shc_cls(self) -> type["SHC"]:
        return self._owner

    def __call__(self, *args, **kwargs):
        raise TypeError(
            f"`{self._class_usage_name}` is a generator-method namespace, "
            "not a standalone constructor. "
            f"Use `{self._class_usage_name}.<method_name>(...)` instead."
        )

    # ============================================================
    # Class generator methods
    # ============================================================

    @generator_method(
        kind="class",
        summary="Generate SHC instance with a given 1D or 2D array. Equivalent to SHC(...).",
        order=10,
    )
    def from_array(
        self,
        cs,
        normalization: str = "4pi",
        dates: Sequence[dt.date] | None = None,
        attrs: dict | None = None,
    ) -> "SHC":
        """
        Generate SHC instance from array.
        """
        cls = self._shc_cls

        return cls(
            _values=cs,
            normalization=normalization,
            dates=dates,
            attrs={} if attrs is None else attrs,
        )

    @generator_method(
        kind="class",
        summary="Generate SHC time series from a trend SHC, for example a GIA model.",
        order=20,
    )
    def from_trend(
        self,
        shc_trend: "SHC",
        dates: Sequence[dt.date],
        ref_time: dt.date | None = None,
    ) -> "SHC":
        """
        Generate SHC time series from a trend SHC.

        C(t) = C_rate * (t - t_ref)
        """
        from sagea.utils import TimeTool

        cls = self._shc_cls

        if len(shc_trend) != 1:
            raise ValueError("shc_trend should contain only one epoch.")

        if ref_time is None:
            ref_time = dates[0]

        year_frac = TimeTool.convert_date_format(
            dates,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.YearFraction,
        )
        year_frac = np.asarray(year_frac)

        ref_year_frac = TimeTool.convert_date_format(
            ref_time,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.YearFraction,
        )

        dt_year = year_frac - ref_year_frac
        trend = shc_trend._values[0]

        values = dt_year[:, None] @ trend[None, :]

        return cls(
            _values=values,
            normalization=shc_trend.normalization,
            dates=dates,
            attrs=shc_trend.attrs.copy(),
        )

    @generator_method(
        kind="class",
        summary="Generate SHC sample(s) from VCM assuming Gaussian distribution.",
        order=30,
    )
    def normal_from_vcm(
        self,
        vcm: np.ndarray,
        nsample: int = 1,
        mean: "SHC | None" = None,
    ) -> "SHC":
        """
        Generate random SHC samples from variance-covariance matrix.
        """
        cls = self._shc_cls

        shape_vcm = vcm.shape

        if len(shape_vcm) != 2 or shape_vcm[0] != shape_vcm[1]:
            raise ValueError(
                f"vcm should be a square 2D matrix, got shape {shape_vcm}."
            )

        ncoef = shape_vcm[0]
        lmax_float = np.sqrt(ncoef) - 1
        lmax = int(round(lmax_float))

        if not np.isclose(lmax_float, lmax, atol=1e-8):
            raise ValueError(
                f"Invalid VCM shape {shape_vcm}. "
                "Expected ncoef = (lmax + 1)^2."
            )

        if mean is None:
            mean_array = np.zeros(ncoef)
        else:
            if not isinstance(mean, cls):
                raise TypeError(f"mean should be {cls.__name__}, got {type(mean)}.")

            if len(mean) != 1:
                raise ValueError("mean SHC should contain only one epoch.")

            if mean.lmax != lmax:
                raise ValueError(
                    f"mean SHC should have lmax={lmax}, got {mean.lmax}."
                )

            mean_array = mean.value[0]

        cs = np.random.multivariate_normal(
            mean=mean_array,
            cov=vcm,
            size=nsample,
        )

        return cls(_values=cs)

    # ============================================================
    # Optional instance generator-like methods
    # ============================================================
    #
    # @generator_method(
    #     kind="instance",
    #     summary="Generate something based on current SHC instance.",
    #     order=100,
    # )
    # def perturb(self, sigma: float) -> "SHC":
    #     shc = self._obj
    #     ...


class SHCGenerateAccessorDescriptor(NamespaceAccessorDescriptor[_SHCGeneratorAccessor]):
    """
    Descriptor for SHC.generate and shc.generate.
    """

    def __init__(self) -> None:
        super().__init__(_SHCGeneratorAccessor)