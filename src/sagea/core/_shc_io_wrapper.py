#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/20 13:35
# @File    : _shc_io_wrapper.py

# !/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Callable, Sequence, TYPE_CHECKING

import numpy as np

from sagea.utils import MathTool
from sagea.core._namespace_accessor import (
    BaseNamespaceAccessor,
    NamespaceAccessorDescriptor,
    namespace_method,
    NamespaceMethodKind,
)

if TYPE_CHECKING:
    from .shc import SHC


def io_method(
        *,
        summary: str | None = None,
        order: int = 0,
        kind: NamespaceMethodKind = "instance",
        show: bool = True,
) -> Callable:
    """
    Mark a method as a public SHC IO method.

    Parameters
    ----------
    summary:
        Short description shown in SHC.io.help() / shc.io.help().
        If None, the first line of the method docstring will be used.
    order:
        Display order in help text.
    kind:
        "class":
            Callable as SHC.io.<method>(...).
        "instance":
            Callable as shc.io.<method>(...).
    show:
        Whether shown in help().
    """
    return namespace_method(
        namespace="io",
        summary=summary,
        order=order,
        kind=kind,
        show=show,
    )


class _SHCIOAccessor(BaseNamespaceAccessor):
    """
    Unified IO accessor for SHC.

    Usage
    -----
    Class methods:
        SHC.io.from_gfc(...)

    Instance methods:
        shc.io.save_file(...)

    Help:
        print(SHC.io.help())
        print(shc.io.help())
    """

    _namespace_name = "io"

    @property
    def _shc_cls(self) -> type["SHC"]:
        return self._owner

    @property
    def _shc(self) -> "SHC":
        if self._obj is None:
            raise RuntimeError("This IO method requires an SHC instance.")
        return self._obj

    # ============================================================
    # Class IO methods
    # ============================================================

    @io_method(
        kind="class",
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

    # ============================================================
    # Instance IO methods
    # ============================================================

    @io_method(
        kind="instance",
        summary="Save a .gfc file for a given-index set of coefficients.",
        order=200,
    )
    def save_file(
            self,
            filepath: str | Path,
            index: int,
            header: str | None = None,
            key: str | None = None,
            overwrite: bool = False,
            make_parent: bool = True,
    ) -> None:
        """
        Save one epoch of SHC coefficients to a text .gfc-like file.
        """

        shc = self._shc

        assert isinstance(filepath, (str, Path))

        if isinstance(filepath, str):
            filepath = Path(filepath)

        save_dir = filepath.parent

        if not save_dir.exists():
            if make_parent:
                save_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"Directory {save_dir} does not exist. "
                    "Use make_parent=True to create it."
                )

        if filepath.exists():
            if overwrite:
                warnings.warn(f"Overwriting {filepath}")
            else:
                raise ValueError(
                    f"Path {filepath} already exists. "
                    "Use overwrite=True to overwrite it."
                )

        if header is None:
            header = ""

        if key is None:
            key = "gfc"

        if index < 0 or index >= shc.ntime:
            raise IndexError(
                f"index {index} is out of range for SHC with ntime={shc.ntime}."
            )

        with open(filepath, "w") as f:
            f.write(header)

            for l in range(shc.lmax + 1):
                for m in range(l + 1):
                    index1d_clm = MathTool.get_cs_1d_index(f"c{l},{m}")
                    index1d_slm = MathTool.get_cs_1d_index(f"s{l},{m}")

                    cvalue = shc.value[index, index1d_clm]
                    cvalue_str = (" " if cvalue >= 0 else "") + f"{cvalue:.12e}"

                    svalue = shc.value[index, index1d_slm]
                    svalue_str = (" " if svalue >= 0 else "") + f"{svalue:.12e}"

                    f.write(
                        f"{key}\t{l}\t{m}\t{cvalue_str}\t{svalue_str}\n"
                    )


class SHCIOAccessorDescriptor(NamespaceAccessorDescriptor[_SHCIOAccessor]):
    """
    Descriptor for SHC.io and shc.io.
    """

    def __init__(self) -> None:
        super().__init__(_SHCIOAccessor)


# ----------------------------------------------------------------------
# Backward-compatible aliases
# ----------------------------------------------------------------------
_SHCClassIOAccessor = _SHCIOAccessor
_SHCInstanceIOAccessor = _SHCIOAccessor
_SHCIODispatcher = SHCIOAccessorDescriptor
