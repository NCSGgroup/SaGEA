#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/22 11:26 
# @File    : _shc_filter_wrapper.py

# !/usr/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/20 13:35
# @File    : _shc_filter_wrapper.py

from __future__ import annotations

import warnings
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .shc import SHC

from sagea.core._namespace_accessor import (
    BaseNamespaceAccessor,
    NamespaceAccessorDescriptor,
    NamespaceMethodKind,
    NamespaceMethodMeta,
    namespace_method,
)

from sagea.filtering.base import (
    cs1d_to_cs2d,
    cs2d_to_cs1d,
)

from sagea.filtering.smoothing import (
    gaussian_filter_cqlm,
    fan_filter_cqlm,
    anisotropic_gaussian_han_filter_cqlm,
)

from sagea.filtering.decorrelation import (
    pnmm_filter_cqlm,
    slide_window_filter_cqlm,
    SlideWindowMode,
)

from sagea.filtering.ddk import apply_ddk_filter_cqlm
from sagea.filtering.regularization import regularization_filter_cs
from sagea.filtering.fsc import fsc_filter_cs

# ----------------------------------------------------------------------
# Backward-compatible type aliases
# ----------------------------------------------------------------------

FilterKind = NamespaceMethodKind
FilterMethodMeta = NamespaceMethodMeta


def filter_method(
        *,
        summary: str | None = None,
        order: int = 0,
        kind: FilterKind = "instance",
        show: bool = True,
) -> Callable:
    """
    Mark a method as a public SHC filter method.

    Parameters
    ----------
    summary:
        Short description shown in SHC.filter.help() / shc.filter.help().
        If None, the first line of the method docstring will be used.
    order:
        Display order in help text.
    kind:
        "instance":
            Callable as shc.filter.<method>(...).

        "class":
            Callable as SHC.filter.<method>(...).
    show:
        Whether this method should be shown in help().
    """
    return namespace_method(
        namespace="filter",
        summary=summary,
        order=order,
        kind=kind,
        show=show,
    )


def instance_filter_method(**kwargs) -> Callable:
    """
    Convenience decorator for instance filter methods.

    Example
    -------
    @instance_filter_method(summary="...", order=10)
    def gaussian(...):
        ...
    """
    kwargs["kind"] = "instance"
    return filter_method(**kwargs)


def class_filter_method(**kwargs) -> Callable:
    """
    Convenience decorator for class filter methods.

    Example
    -------
    @class_filter_method(summary="...", order=10)
    def available(...):
        ...
    """
    kwargs["kind"] = "class"
    return filter_method(**kwargs)


class _SHCFilterAccessor(BaseNamespaceAccessor):
    """
    Accessor namespace for SHC filter methods.

    Recommended usage
    -----------------
    Instance filter methods:

        shc.filter.gaussian(...)
        shc.filter.ddk(...)

    Class filter methods:

        SHC.filter.available(...)
        SHC.filter.help()

    Help
    ----
    Both are valid:

        print(SHC.filter.help())
        print(shc.filter.help())

    Deprecated but temporarily supported
    ------------------------------------
    Old dispatcher style:

        shc.filter("gaussian", ...)
    """

    _namespace_name = "filter"

    _METHOD_MAP: dict[str, str] = {
        # Old name -> new accessor method name.
        "gaussian": "gaussian",
        "fan": "fan",
        "han": "han",
        "ddk": "ddk",
        "regularization": "regularization",
        "pnmm": "pnmm",
        "slidewindowstable": "slidewindowStable",
        "slidewindowswenson2006": "slidewindowSwenson2006",
        "slidewindowduan2009": "slidewindowDuan2009",
        "fsc": "fsc",
    }

    @property
    def _shc(self) -> "SHC":
        """
        Return current SHC instance.

        Raises
        ------
        RuntimeError
            If current accessor is class-context, i.e. SHC.filter.
        """
        if self._obj is None:
            raise RuntimeError(
                "This filter method requires an SHC instance. "
                f"Use `{self._instance_usage_name}.<method_name>(...)`."
            )

        return self._obj

    # ============================================================
    # Deprecated call-style interface
    # ============================================================

    def __call__(
            self,
            name: Any | None = None,
            /,
            *args: Any,
            **kwargs: Any,
    ) -> "SHC":
        """
        Deprecated dispatcher.

        Old style
        ---------
        shc.filter("method_name", ...)

        New style
        ---------
        shc.filter.method_name(...)

        Notes
        -----
        This method is kept only for backward compatibility.
        """

        if self._obj is None:
            raise TypeError(
                f"`{self._class_usage_name}` is a filter-method namespace, "
                "not a standalone operation. "
                f"Use `{self._instance_usage_name}.<method_name>(...)` on an SHC instance."
            )

        if name is None:
            raise TypeError(
                f"`{self._instance_usage_name}` is a filter-method namespace, "
                "not a standalone operation. "
                f"Use `{self._instance_usage_name}.<method_name>(...)` instead."
            )

        try:
            from .shc import SHCDeprecationWarning
        except Exception:
            SHCDeprecationWarning = FutureWarning

        if isinstance(name, str):
            new_method_name = self._METHOD_MAP.get(name.lower())

            if new_method_name:
                message = (
                    f"`shc.filter({name!r}, ...)` is deprecated and will be "
                    f"removed in a future version. "
                    f"Use `shc.filter.{new_method_name}(...)` instead."
                )
            else:
                message = (
                    f"`shc.filter({name!r}, ...)` is deprecated and will be "
                    f"removed in a future version. "
                    "Use `shc.filter.<method_name>(...)` instead."
                )
        else:
            message = (
                "`shc.filter(method, ...)` is deprecated and will be "
                "removed in a future version. "
                "Use `shc.filter.<method_name>(...)` instead."
            )

        warnings.warn(
            message,
            category=SHCDeprecationWarning,
            stacklevel=2,
        )

        return self._shc._old_filter(name, *args, **kwargs)

    # ============================================================
    # Shared finalize logic
    # ============================================================

    def _finalize(self, new_coeffs, *, inplace: bool = False) -> "SHC":
        """
        Finalize filtered coefficients.

        Parameters
        ----------
        new_coeffs:
            Filtered SHC coefficients.
        inplace:
            If True, modify current SHC object in-place.

        Returns
        -------
        SHC
            Filtered SHC object.
        """
        shc = self._shc

        if inplace:
            shc.value = new_coeffs
            return shc

        new = shc.copy()
        new.value = new_coeffs
        return new

    # ============================================================
    # Class filter methods
    # ============================================================

    @class_filter_method(
        summary="Return registered filter methods grouped by method kind.",
        order=0,
    )
    def available(self) -> dict[str, list[str]]:
        """
        Return all registered filter methods grouped by kind.

        Returns
        -------
        dict
            Example:

            {
                "class": ["available"],
                "instance": ["gaussian", "fan", "ddk", ...],
            }
        """
        return {
            "class": [
                name
                for name, _, _ in self._iter_namespace_methods(kind="class")
            ],
            "instance": [
                name
                for name, _, _ in self._iter_namespace_methods(kind="instance")
            ],
        }

    # ============================================================
    # Instance filter methods
    # ============================================================
    #
    # Each real method should:
    #
    # 1. Have an explicit signature.
    # 2. Put method-specific parameters first.
    # 3. Put common keyword-only parameters such as inplace at the end.
    # 4. Return SHC.
    #
    # Template:
    #
    # @filter_method(summary="...", order=...)
    # def your_method(
    #     self,
    #     required_arg,
    #     *,
    #     optional_arg=None,
    #     inplace: bool = False,
    # ) -> "SHC":
    #     ...
    #     new_coeffs = ...
    #     return self._finalize(new_coeffs, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "Gaussian filter, see Wahr et al., 1998, "
                "https://doi.org/10.1029/98JB02844"
        ),
        order=199800,
    )
    def gaussian(self, radius: int, *, inplace: bool = False) -> "SHC":
        """
        Apply Gaussian smoothing filter.

        Parameters
        ----------
        radius:
            Smoothing radius in kilometers.
        inplace:
            Whether to modify current SHC object in-place.

        Returns
        -------
        SHC
            Filtered SHC object.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = gaussian_filter_cqlm(
            cqlm,
            sqlm,
            lmax=shc.lmax,
            radius_km=radius,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "Fan filter designed by Zhang et al., 2009, "
                "https://doi.org/10.1029/2009GL039459"
        ),
        order=200900,
    )
    def fan(
            self,
            radius1: int,
            radius2: int,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply Fan filter.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = fan_filter_cqlm(
            cqlm,
            sqlm,
            lmax=shc.lmax,
            radius1_km=radius1,
            radius2_km=radius2,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "A non-isotropic filter designed by Han et al., 2005, "
                "https://doi.org/10.1111/j.1365-246X.2005.02756.x"
        ),
        order=200500,
    )
    def han(
            self,
            radius1: int,
            radius2: int,
            m0: int,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply anisotropic Gaussian Han filter.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = anisotropic_gaussian_han_filter_cqlm(
            cqlm,
            sqlm,
            lmax=shc.lmax,
            radius1_km=radius1,
            radius2_km=radius2,
            m0=m0,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "DDK filter designed by Kusche, 2007, "
                "https://doi.org/10.1007/s00190-007-0143-3"
        ),
        order=200700,
    )
    def ddk(
            self,
            ddk_id: int,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply DDK filter.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = apply_ddk_filter_cqlm(
            cqlm,
            sqlm,
            ddk_type=ddk_id,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "A regularization filter with selected covariance matrix, "
                "see Kusche, 2007, https://doi.org/10.1007/s00190-007-0143-3"
        ),
        order=200701,
    )
    def regularization(
            self,
            vcm_err,
            vcm_sig,
            from_degree=2,
            alpha=1,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply regularization filter.
        """

        shc = self._shc

        cs_f = regularization_filter_cs(
            cs=shc.value,
            vcm_err=vcm_err,
            vcm_sig=vcm_sig,
            alpha=alpha,
            from_degree=from_degree,
        )

        return self._finalize(cs_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "An empirical de-correlation filter based on polynomial fitting, "
                "see Chambers, 2006, https://doi.org/10.1029/2006GL027296"
        ),
        order=200603,
    )
    def pnmm(
            self,
            n: int,
            m: int,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply PnMm de-correlation filter.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = pnmm_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=n,
            start_m=m,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "An empirical de-correlation filter based on polynomial fitting "
                "in a moving window, see Swenson and Wahr, 2006, "
                "https://doi.org/10.1029/2005GL025285"
        ),
        order=200601,
    )
    def slidewindowSwenson2006(
            self,
            n: int,
            m: int,
            a: int,
            k: int,
            window_length_min: int,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply Swenson and Wahr 2006 moving-window de-correlation filter.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = slide_window_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=n,
            start_m=m,
            window_length=window_length_min,
            mode=SlideWindowMode.Wahr2006,
            a=a,
            k=k,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "An empirical de-correlation filter based on polynomial fitting "
                "in a moving window, see Duan et al., 2009, "
                "https://doi.org/10.1007/s00190-009-0327-0"
        ),
        order=200901,
    )
    def slidewindowDuan2009(
            self,
            n: int,
            m: int,
            a: int,
            k: int,
            window_length_min: int,
            gamma: float,
            p: int,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply Duan et al. 2009 moving-window de-correlation filter.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = slide_window_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=n,
            start_m=m,
            window_length=window_length_min,
            mode=SlideWindowMode.Duan2009,
            a=a,
            k=k,
            gamma=gamma,
            p=p,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "An empirical de-correlation filter based on polynomial fitting "
                "in a fixed-length moving window, see Swenson and Wahr, 2006, "
                "https://doi.org/10.1029/2005GL025285"
        ),
        order=200602,
    )
    def slidewindowStable(
            self,
            n: int,
            m: int,
            window_length: int,
            *,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply stable fixed-length moving-window de-correlation filter.
        """

        shc = self._shc

        cqlm, sqlm, single = cs1d_to_cs2d(shc.value)

        cqlm_f, sqlm_f = slide_window_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=n,
            start_m=m,
            window_length=window_length,
            mode=SlideWindowMode.Stable,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        kind="instance",
        summary=(
                "A regularization filter with a hybrid signal VCM, "
                "see Liu et al., https://doi.org/under.submission"
        ),
        order=200902,
    )
    def fsc(
            self,
            vcm_err,
            vcm_sig_list,
            init_alphas,
            from_degree,
            scale,
            *,
            tol=1e-4,
            max_iter=100,
            verbose=True,
            inplace: bool = False,
    ) -> "SHC":
        """
        Apply FSC filter.
        """

        shc = self._shc

        cs_f = fsc_filter_cs(
            cs=shc.value,
            vcm_err=vcm_err,
            vcm_sig_list=vcm_sig_list,
            init_alphas=init_alphas,
            from_degree=from_degree,
            scale=scale,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose,
        )

        return self._finalize(cs_f, inplace=inplace)


class SHCFilterAccessorDescriptor(
    NamespaceAccessorDescriptor[_SHCFilterAccessor]
):
    """
    Descriptor for SHC.filter and shc.filter.
    """

    def __init__(self) -> None:
        super().__init__(_SHCFilterAccessor)
