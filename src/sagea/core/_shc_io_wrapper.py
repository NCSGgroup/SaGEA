#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/20 13:35
# @File    : _shc_io_wrapper.py

from __future__ import annotations

import datetime
import inspect
from pathlib import Path
from typing import Any, Callable, Iterator, TYPE_CHECKING, Sequence

import numpy as np
from sagea.utils import MathTool

if TYPE_CHECKING:
    from .shc import SHC


def io_method(
        *,
        summary: str | None = None,
        order: int = 0,
) -> Callable:
    """
    Mark a method as a public SHC io method.
    Parameters
    ----------
    summary:
        Short description shown in shc.io.help().
        If None, the first line of the method docstring will be used.
    order:
        Display order in help text.
    """

    def decorator(func: Callable) -> Callable:
        func._is_shc_io_method = True
        func._shc_io_summary = summary
        func._shc_io_order = order
        return func

    return decorator


class _SHCClassIOAccessor:
    """
    Accessor namespace for SHC IO methods.

    Usage
    -----------
    SHC.io.from_xxx(...)
    """

    def __init__(self, shc_cls: type["SHC"]) -> None:
        self._shc_cls = shc_cls

    # ============================================================
    # Public display / help API
    # ============================================================

    def __str__(self) -> str:
        return self._format_methods(verbose=False)

    def __repr__(self) -> str:
        method_names = [name for name, _ in self._iter_io_methods()]
        return (
            f"<{type(self).__name__} "
            f"available_methods={method_names}>"
        )

    def __help__(self) -> str:
        return self._format_methods(verbose=True)

    def help(self) -> str:
        return self.__help__()

    def __dir__(self) -> list[str]:
        default_attrs = set(super().__dir__())
        method_names = {name for name, _ in self._iter_io_methods()}
        return sorted(default_attrs | method_names)

    # ============================================================
    # Internal method discovery
    # ============================================================
    @classmethod
    def _iter_io_methods(cls) -> Iterator[tuple[str, Callable]]:
        methods: list[tuple[str, Callable]] = []
        for name, obj in inspect.getmembers(cls, predicate=callable):
            if getattr(obj, "_is_shc_io_method", False):
                methods.append((name, obj))
        methods.sort(
            key=lambda item: (
                getattr(item[1], "_shc_io_order", 0),
                item[0],
            )
        )
        yield from methods

    @staticmethod
    def _method_signature(func: Callable) -> str:
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if params and params[0].name in {"self", "cls"}:
            params = params[1:]
        new_sig = sig.replace(parameters=params)
        return str(new_sig)

    @staticmethod
    def _method_summary(func: Callable) -> str:
        summary = getattr(func, "_shc_io_summary", None)
        if summary:
            return summary
        doc = inspect.getdoc(func)
        if not doc:
            return ""
        return doc.strip().splitlines()[0]

    def _format_methods(self, *, verbose: bool) -> str:
        methods = list(self._iter_io_methods())
        if not methods:
            return (
                "No public IO methods are currently registered.\n"
                "Please decorate IO methods with @io_method(...)."
            )
        lines: list[str] = []
        lines.append("Available SHC IO methods:")
        lines.append("")
        for name, func in methods:
            signature = self._method_signature(func)
            summary = self._method_summary(func)
            if verbose:
                lines.append(f"  SHC.io.{name}{signature}")
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
        lines.append("  SHC.io.<method_name>(...)")
        return "\n".join(lines)

    # ============================================================
    # IO methods (examples)
    # ============================================================

    @io_method(
        summary="Read SHC from one or multiple .gfc files.",
        order=100,
    )
    def from_gfc(
            self,
            filepath: str | Path | Sequence[str | Path],
            lmax: int,
            key: str = "gfc",
            cols=None,
            normalization: str = "4pi",
            dates: Sequence[datetime.date] | None = None,
            attrs: dict | None = None,
    ) -> "SHC":
        """
        Read SHC from one or multiple GFC files.
        """
        from sagea.sgio._gfc_reader import read_gfc

        cls = self._shc_cls

        if isinstance(filepath, (str, Path)):
            cs = read_gfc(
                Path(filepath),
                key=key,
                lmax=lmax,
                col_indices=cols,
            )
        else:
            assert len(filepath) >= 1

            cs = np.asarray(
                [
                    read_gfc(
                        Path(path),
                        key=key,
                        lmax=lmax,
                        col_indices=cols,
                    )
                    for path in filepath
                ]
            )

        return cls(
            _values=cs,
            normalization=normalization,
            dates=dates,
            attrs={} if attrs is None else attrs,
        )


class _SHCInstanceIOAccessor:
    """
    IO accessor for SHC instance.

    Used as:
        shc.io.save_gfc(...)
        shc.io.to_file(...)
    """

    def __init__(self, shc: "SHC") -> None:
        self._shc = shc

    @io_method(
        summary="Save a .gfc file for a given-index set of coefficients.",
        order=200,
    )
    def save_file(self, filepath: str | Path, index: int, header: str | None = None, key: str | None = None) -> None:

        assert isinstance(filepath, (str, Path))
        if isinstance(filepath, str):
            filepath = Path(filepath)

        save_dir = filepath.parent
        if not save_dir.exists():
            save_dir.mkdir()

        if filepath.exists():
            raise ValueError(f"{filepath} already exists.")

        if header is None:
            header = ""
        if key is None:
            key = "gfc"

        with open(filepath, 'w') as f:
            f.write(header)

            for l in range(self._shc.lmax + 1):
                for m in range(l + 1):
                    index1d_clm = MathTool.get_cs_1d_index(f"c{l},{m}")
                    index1d_slm = MathTool.get_cs_1d_index(f"s{l},{m}")

                    cvalue = self._shc.value[index, index1d_clm]
                    cvalue_str = (" " if cvalue >= 0 else "") + f"{cvalue:.12e}"

                    svalue = self._shc.value[index, index1d_slm]
                    svalue_str = (" " if svalue >= 0 else "") + f"{svalue:.12e}"

                    f.write(f"{key}\t{l}\t{m}\t{cvalue_str}\t{svalue_str}\n")


class _SHCIODispatcher:
    """
    Descriptor dispatching SHC.io and shc.io.

    SHC.io  -> _SHCClassIOAccessor
    shc.io  -> _SHCInstanceIOAccessor
    """

    def __set_name__(self, owner, name):
        self.name = name
        self.cache_name = f"_{name}_class_accessor"

    def __get__(self, obj, owner=None):
        if owner is None:
            owner = type(obj)

        # SHC.io
        if obj is None:
            accessor = owner.__dict__.get(self.cache_name)
            if accessor is None:
                accessor = _SHCClassIOAccessor(owner)
                setattr(owner, self.cache_name, accessor)
            return accessor

        # shc.io
        return _SHCInstanceIOAccessor(obj)
