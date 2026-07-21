#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/20 15:31 
# @File    : _shc_generator_wrapper.py

from __future__ import annotations
from typing import Any, Callable, Iterator, Sequence, TYPE_CHECKING

from statsmodels.genmod.families.links import sqrt

if TYPE_CHECKING:
    from .shc import SHC

import datetime as dt
import inspect
import weakref
from pathlib import Path

import numpy as np


def generator_method(
        *,
        summary: str | None = None,
        order: int = 0,
) -> Callable:
    """
    Mark a method as a public SHC generator method.

    Parameters
    ----------
    summary:
        Short description shown in SHC.generate.help().
        If None, the first line of the method docstring will be used.
    order:
        Display order in help text.
    """

    def decorator(func: Callable) -> Callable:
        func._is_shc_generator_method = True
        func._shc_generator_summary = summary
        func._shc_generator_order = order
        return func

    return decorator


class _SHCGeneratorAccessor:
    """
    Class-level accessor namespace for SHC generator methods.

    Recommended usage
    -----------------
    SHC.generate.from_gfc(...)
    SHC.generate.from_trend(...)

    Invalid usage
    -------------
    shc.generate.from_gfc(...)
    """

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        names.update(name for name, _ in self._iter_generator_methods())
        return sorted(names)

    def __init__(self, shc_cls: type["SHC"]) -> None:
        self._shc_cls = shc_cls

    def __call__(self, *args: Any, **kwargs: Any) -> "SHC":
        """
        Do not allow SHC.generate(...) directly.

        Users should call:

            SHC.generate.<method_name>(...)
        """
        raise TypeError(
            "`SHC.generate` is a generator-method namespace, not a standalone "
            "constructor. Use `SHC.generate.<method_name>(...)` instead."
        )

    # ============================================================
    # Display / help API
    # ============================================================

    def __str__(self) -> str:
        """
        Return a concise string showing available generator methods.

        Example
        -------
        print(SHC.generate)
        """
        return self._format_methods(verbose=False)

    def __repr__(self) -> str:
        method_names = [name for name, _ in self._iter_generator_methods()]
        return (
            f"<{type(self).__name__} "
            f"owner={self._shc_cls.__name__!r} "
            f"available_methods={method_names}>"
        )

    def __help__(self) -> str:
        """
        Non-standard helper.

        Note
        ----
        Python built-in help(obj) does not automatically call __help__().
        Prefer using:

            print(SHC.generate.help())
        """
        return self._format_methods(verbose=True)

    def help(self) -> str:
        """
        Return detailed help text for available generator methods.

        Usage
        -----
        print(SHC.generate.help())
        """
        return self.__help__()

    def __dir__(self) -> list[str]:
        """
        Improve interactive auto-completion.

        This makes dir(SHC.generate) include registered generator method names.
        """
        default_attrs = set(super().__dir__())
        method_names = {name for name, _ in self._iter_generator_methods()}
        return sorted(default_attrs | method_names)

    # ============================================================
    # Internal method discovery
    # ============================================================

    @classmethod
    def _iter_generator_methods(cls) -> Iterator[tuple[str, Callable]]:
        """
        Iterate over methods decorated by @generator_method.
        """
        methods: list[tuple[str, Callable]] = []

        for name, obj in inspect.getmembers(cls, predicate=callable):
            if getattr(obj, "_is_shc_generator_method", False):
                methods.append((name, obj))

        methods.sort(
            key=lambda item: (
                getattr(item[1], "_shc_generator_order", 0),
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
        summary = getattr(func, "_shc_generator_summary", None)

        if summary:
            return summary

        doc = inspect.getdoc(func)

        if not doc:
            return ""

        return doc.strip().splitlines()[0]

    def _format_methods(self, *, verbose: bool) -> str:
        """
        Format available generator methods.
        """
        methods = list(self._iter_generator_methods())

        if not methods:
            return (
                "No public SHC generator methods are currently registered.\n"
                "Please decorate generator methods with @generator_method(...)."
            )

        lines: list[str] = []

        lines.append("Available SHC generator methods:")
        lines.append("")

        for name, func in methods:
            signature = self._method_signature(func)
            summary = self._method_summary(func)

            if verbose:
                lines.append(f"  {self._shc_cls.__name__}.generate.{name}{signature}")
                if summary:
                    lines.append(f"      {summary}")
                lines.append("")
            else:
                if summary:
                    lines.append(f"  - {name}{signature}: {summary}")
                else:
                    lines.append(f"  - {name}{signature}")

        lines.append("")
        lines.append("Recommended usage:")
        lines.append(f"  {self._shc_cls.__name__}.generate.<method_name>(...)")

        lines.append("")
        lines.append("Note:")
        lines.append(
            f"  `{self._shc_cls.__name__}.generate` is only available from the class, "
            "not from an instance."
        )

        return "\n".join(lines)

    # ============================================================
    # Constructors / generators
    # ============================================================
    @generator_method(
        summary="Generate SHC instance with a given 1d- or 2d-array. Equivalent to SHC(...)",
        order=10,
    )
    def from_array(self, cs):
        cls = self._shc_cls
        return cls(cs)

    @generator_method(
        summary="Generate SHC time series from a trend SHC (a GIA model, for example).",
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
        summary="Generate SHC sample(s) from VCM assuming Gaussian distribution.",
        order=30,
    )
    def normal_from_vcm(
            self,
            vcm: np.ndarray,
            nsample: int = 1,
            mean: "SHC" = None
    ) -> "SHC":

        cls = self._shc_cls

        shape_vcm = vcm.shape
        assert len(shape_vcm) == 2
        square_root = np.sqrt(shape_vcm[0])
        assert square_root % 1 < 1e-8, f"invalid shape of input vcm {shape_vcm}."

        lmax = square_root - 1
        if mean is None:
            mean = np.zeros(shape_vcm[:1])
        else:
            assert isinstance(mean, cls)
            assert len(mean) == 1, "mean SHC should have only one element."
            assert mean.lmax == lmax, f"mean SHC should have a same lmax as vcm {lmax}, got {mean.lmax} instead."

            mean = mean.value[0]

        cs = np.random.multivariate_normal(mean=mean, cov=vcm, size=nsample)
        return cls(cs)
