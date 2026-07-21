#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/20 13:35 
# @File    : _shc_filter_wrapper.py

from __future__ import annotations
import inspect
import warnings
from typing import Any, Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .shc import SHC

import numpy as np

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


def filter_method(
        *,
        summary: str | None = None,
        order: int = 0,
) -> Callable:
    """
    Mark a method as a public SHC filter method.
    Parameters
    ----------
    summary:
        Short description shown in shc.filter.help().
        If None, the first line of the method docstring will be used.
    order:
        Display order in help text.
    """

    def decorator(func: Callable) -> Callable:
        func._is_shc_filter_method = True
        func._shc_filter_summary = summary
        func._shc_filter_order = order
        return func

    return decorator


class _SHCFilterAccessor:
    """
    Accessor namespace for SHC filter methods.

    Usage
    -----------
    shc.filter.some_method(...)

    Deprecated but temporarily supported
    ------------------------------------
    shc.filter("some_method", ...)
    """

    _METHOD_MAP: dict[str, str] = {
        # Example format:
        # "old_name": "new_method_name",
        #
        # If old and new names are identical:
        # "method_name": "method_name",
        "gaussian": "gaussian",
        "fan": "fan",
        "han": "han",
        "ddk": "ddk",
        "regularization": "regularization",
        "pnmm": "pnmm",
        "slidewindowstable": "",
        "slidewindowswenson2006": "slidewindowswenson2006",
        "slidewindowduan2009": "slidewindowduan2009",
        "fsc": "fsc",
    }

    def __init__(self, shc: "SHC") -> None:
        self._shc = shc

    def __call__(
            self,
            name: str | None = None,
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
        """
        from core.shc import SHCDeprecationWarning

        if name is None:
            raise TypeError(
                "`shc.filter` is a filter-method namespace, not a standalone "
                "operation. Use `shc.filter.<method_name>(...)` instead."
            )
        else:
            new_method_name = self._METHOD_MAP[name]
            warnings.warn(
                (
                    f"`shc.filter({name!r}, ...)` is deprecated from v0.2.9 and will be "
                    f"removed in a future version. "
                    f"Use `shc.filter.{new_method_name}(...)` instead. "
                    f"See README or documentation for migration details."
                ),
                category=SHCDeprecationWarning,
                stacklevel=2,
            )
            return self._shc._old_filter(name, *args, **kwargs)

    # ============================================================
    # Public display / help API
    # ============================================================

    def __str__(self) -> str:
        """
        Return a concise string showing available filter methods.
        Example
        -------
        print(shc.filter)
        """
        return self._format_methods(verbose=False)

    def __repr__(self) -> str:
        method_names = [name for name, _ in self._iter_filter_methods()]
        return (
            f"<{type(self).__name__} "
            f"available_methods={method_names}>"
        )

    def __help__(self) -> str:
        """
        Non-standard helper.
        Note
        ----
        Python built-in help(obj) does not automatically call __help__().
        Prefer using shc.filter.help().
        """
        return self._format_methods(verbose=True)

    def help(self) -> str:
        """
        Return detailed help text for available filter methods.
        Usage
        -----
        print(shc.filter.help())
        """
        return self.__help__()

    def __dir__(self) -> list[str]:
        """
        Improve interactive auto-completion.
        This makes dir(shc.filter) include registered filter method names.
        """
        default_attrs = set(super().__dir__())
        method_names = {name for name, _ in self._iter_filter_methods()}
        return sorted(default_attrs | method_names)

    # ============================================================
    # Internal method discovery
    # ============================================================
    @classmethod
    def _iter_filter_methods(cls) -> Iterator[tuple[str, Callable]]:
        """
        Iterate over methods decorated by @filter_method.
        """
        methods: list[tuple[str, Callable]] = []
        for name, obj in inspect.getmembers(cls, predicate=callable):
            if getattr(obj, "_is_shc_filter_method", False):
                methods.append((name, obj))
        methods.sort(
            key=lambda item: (
                getattr(item[1], "_shc_filter_order", 0),
                item[0],
            )
        )
        yield from methods

    @staticmethod
    def _method_signature(func: Callable) -> str:
        """
        Return a readable method signature without `self`.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        new_sig = sig.replace(parameters=params)
        return str(new_sig)

    @staticmethod
    def _method_summary(func: Callable) -> str:
        """
        Get method summary from decorator or docstring.
        """
        summary = getattr(func, "_shc_filter_summary", None)
        if summary:
            return summary
        doc = inspect.getdoc(func)
        if not doc:
            return ""
        return doc.strip().splitlines()[0]

    def _format_methods(self, *, verbose: bool) -> str:
        """
        Format available filter methods.
        """
        methods = list(self._iter_filter_methods())
        if not methods:
            return (
                "No public filter methods are currently registered.\n"
                "Please decorate filter methods with @filter_method(...)."
            )
        lines: list[str] = []
        lines.append("Available SHC filter methods:")
        lines.append("")
        for name, func in methods:
            signature = self._method_signature(func)
            summary = self._method_summary(func)
            if verbose:
                lines.append(f"  shc.filter.{name}{signature}")
                if summary:
                    lines.append(f"      {summary}")
                lines.append("")
            else:
                if summary:
                    lines.append(f"  - {name}{signature}: {summary}")
                else:
                    lines.append(f"  - {name}{signature}")
        lines.append("")
        lines.append("Usage:")
        lines.append("  shc.filter.<method_name>(...)")
        if self._METHOD_MAP:
            lines.append("")
            lines.append("Deprecated old-style usage is still supported temporarily:")
            lines.append('  shc.filter("method_name", ...)')
            lines.append("")
            lines.append("Please check README or documentation for migration details.")
        return "\n".join(lines)

    # ============================================================
    # Shared finalize logic
    # ============================================================

    def _finalize(
            self,
            new_coeffs: np.ndarray,
            *,
            inplace: bool,
    ) -> "SHC":
        """
        Common inplace / non-inplace behavior.
        """
        if inplace:
            self._shc.value[...] = new_coeffs
            return self._shc

        new_shc = self._shc.copy()
        new_shc.value = np.asarray(new_coeffs)
        return new_shc

    # ============================================================
    # filter method
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
        summary="Gaussian filter, see Wahr et al., 1998, https://doi.org/10.1029/98JB02844",
        order=199800,
    )
    def gaussian(self, radius: int, *, inplace: bool = False) -> "SHC":

        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

        cqlm_f, sqlm_f = gaussian_filter_cqlm(
            cqlm,
            sqlm,
            lmax=self._shc.lmax,
            radius_km=radius,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        summary="Fan filter designed by Zhang et al., 2009, https://doi.org/10.1029/2009GL039459",
        order=200900,
    )
    def fan(self, radius1: int, radius2: int, *, inplace: bool = False) -> "SHC":

        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

        cqlm_f, sqlm_f = fan_filter_cqlm(
            cqlm,
            sqlm,
            lmax=self._shc.lmax,
            radius1_km=radius1,
            radius2_km=radius2,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        summary="A non-isotropic filter designed by Han et al., 2005, https://doi.org/10.1111/j.1365-246X.2005.02756.x",
        order=200500,
    )
    def han(self, radius1: int, radius2: int, m0: int, *, inplace: bool = False) -> "SHC":
        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

        cqlm_f, sqlm_f = anisotropic_gaussian_han_filter_cqlm(
            cqlm,
            sqlm,
            lmax=self._shc.lmax,
            radius1_km=radius1,
            radius2_km=radius2,
            m0=m0
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        summary="DDK filter designed by Kusche, 2007, https://doi.org/10.1007/s00190-007-0143-3",
        order=200700,
    )
    def ddk(self, ddk_id: int, *, inplace: bool = False) -> "SHC":
        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

        cqlm_f, sqlm_f = apply_ddk_filter_cqlm(
            cqlm,
            sqlm,
            ddk_type=ddk_id,
        )

        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        summary="A regularization filter with selected the covariance matrix, see Kusche, 2007, https://doi.org/10.1007/s00190-007-0143-3",
        order=200701,
    )
    def regularization(self, vcm_err, vcm_sig, from_degree=2, alpha=1, *, inplace: bool = False) -> "SHC":
        cs = self._shc.value
        cs_f = regularization_filter_cs(
            cs=cs,
            vcm_err=vcm_err,
            vcm_sig=vcm_sig,
            alpha=alpha,
            from_degree=from_degree,
        )

        return self._finalize(cs_f, inplace=inplace)

    @filter_method(
        summary="An empirical de-correlation filter based on polynomial fitting, see Chambers, 2006, https://doi.org/10.1029/2006GL027296",
        order=200603,
    )
    def pnmm(self, n: int, m: int, *, inplace: bool = False) -> "SHC":
        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

        cqlm_f, sqlm_f = pnmm_filter_cqlm(
            cqlm,
            sqlm,
            poly_n=n,
            start_m=m,
        )
        cs1d_f = cs2d_to_cs1d(cqlm_f, sqlm_f, single=single)
        return self._finalize(cs1d_f, inplace=inplace)

    @filter_method(
        summary="An empirical de-correlation filter based on polynomial fitting in a moving window, see Swenson and Wahr, 2006, https://doi.org/10.1029/2005GL025285",
        order=200601,
    )
    def slidewindowSwenson2006(self, n: int, m: int, a: int, k: int, window_length_min: int, *,
                               inplace: bool = False) -> "SHC":
        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

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
        summary="An empirical de-correlation filter based on polynomial fitting in a moving window, see Duan et al., 2009, https://doi.org/10.1007/s00190-009-0327-0",
        order=200901,
    )
    def slidewindowDuan2009(self, n: int, m: int, a: int, k: int, window_length_min: int, gamma: float, p: int, *,
                            inplace: bool = False) -> "SHC":
        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

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
        summary="An empirical de-correlation filter based on polynomial fitting in a fixes-length moving window, see Swenson and Wahr, 2006, https://doi.org/10.1029/2005GL025285",
        order=200602,
    )
    def slidewindowStable(self, n: int, m: int, window_length: int, *, inplace: bool = False) -> "SHC":
        cqlm, sqlm, single = cs1d_to_cs2d(self._shc.value)

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
        summary="A regularization filter with a hybrid signal VCM, see Liu et al., https://doi.org/under.submission",
        order=200602,
    )
    def fsc(
            self, vcm_err, vcm_sig_list, init_alphas, from_degree, scale, *,
            tol=1e-4,
            max_iter=100,
            verbose=True,
            inplace: bool = False
    ) -> "SHC":

        cs = self._shc.value
        cs_f = fsc_filter_cs(
            cs=cs,
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
