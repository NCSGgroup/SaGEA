#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/8/10 11:35 
# @File    : _shc_plot_wrapper.py

from __future__ import annotations

import warnings
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


def shc_plot_method(
        *,
        summary: str | None = None,
        order: int = 0,
        kind: NamespaceMethodKind = "instance",
        show: bool = True,
) -> Callable:
    """
    Mark a method as a public SHC plot method.

    Parameters
    ----------
    summary:
        Short description shown in SHC.plot.help() / shc.plot.help().
        If None, the first line of the method docstring will be used.
    order:
        Display order in help text.
    kind:
        "class":
            Callable as SHC.plot.<method>(...).
        "instance":
            Callable as shc.plot.<method>(...).
    show:
        Whether shown in help().
    """
    return namespace_method(
        namespace="plot",
        summary=summary,
        order=order,
        kind=kind,
        show=show,
    )


class _SHCPlotAccessor(BaseNamespaceAccessor):
    """
    Unified Plot accessor for SHC.

    Usage
    -----
    Class methods:
        /

    Instance methods:
        SHC.plot.triangle(...)
        shc.plot.drms(...)
        shc.plot.drss(...)
        shc.plot.drss_cumulative(...)

    Help:
        print(SHC.plot.help())
        print(shc.plot.help())
    """

    _namespace_name = "plot"

    @property
    def _shc_cls(self) -> type["SHC"]:
        return self._owner

    @property
    def _shc(self) -> "SHC":
        if self._obj is None:
            raise RuntimeError("This Plot method requires an SHC instance.")
        return self._obj

    # ============================================================
    # Class IO methods
    # ============================================================

    # ============================================================
    # Instance IO methods
    # ============================================================

    @shc_plot_method(
        kind="instance",
        summary="2D plot of the spherical harmonic spectrum",
        order=200,
    )
    def triangle(
            self,
            vmin=None,
            vmax=None,
            cbar=True,
            cbar_label=None,
            index=None,
            figsize=None,
            title=None,
            subtitles=None,
            cmap="RdBu_r",
            log10_absolute=False,
            max_plots=24,
            ncols=None,
            ignore_degrees=None,
            savepath=None,
            dpi=300,
            show=True,
            **matshow_kwargs,
    ):
        """
        2D plot of the spherical harmonic spectrum.
        """

        import math
        import matplotlib.pyplot as plt

        shc = self._shc

        # ----------------------------
        # 1. Select indices
        # ----------------------------
        n_total = len(shc)

        if index is None:
            indices = list(range(n_total))
        elif isinstance(index, int):
            indices = [index]
        else:
            indices = list(index)

        if len(indices) == 0:
            raise ValueError("No grid index selected.")

        if len(indices) > max_plots:
            warnings.warn(
                f"Too many grids to plot: {len(indices)}. "
                f"Show only the first {max_plots} indices/grids."
            )
            indices = indices[:max_plots]

        for idx in indices:
            if idx < 0 or idx >= n_total:
                raise IndexError(f"Grid index out of range: {idx}")

        data = shc.value[indices]
        nplot = len(indices)

        # ----------------------------
        # 2. Shared colorbar logic
        # ----------------------------
        shared_colorbar = not (vmin is None and vmax is None)

        if shared_colorbar:
            if vmin is None:
                vmin = np.nanmin(data)
            if vmax is None:
                vmax = np.nanmax(data)

        # ----------------------------
        # 3. Layout
        # ----------------------------
        if ncols is None:
            if nplot == 1:
                ncols = 1
            elif nplot <= 4:
                ncols = 2
            else:
                ncols = 3

        ncols = int(ncols)
        nrows = math.ceil(nplot / ncols)

        if figsize is None:
            figsize = (4.5 * ncols, 3.4 * nrows)

        # ----------------------------
        # 5. Create figure
        # ----------------------------
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=False,
        )

        axes_flat = axes.ravel()
        last_mappable = None

        # ----------------------------
        # 6. Plot each panel
        # ----------------------------

        cqlm, sqlm = shc.cs2d

        if ignore_degrees is not None:
            for degree in ignore_degrees:
                cqlm[:, degree, :], sqlm[:, degree, :] = np.nan, np.nan

        for k, idx in enumerate(indices):

            plot_mat = np.full((shc.lmax + 1, 2 * shc.lmax + 1), np.nan)
            for l in range(shc.lmax + 1):
                for m in range(l + 1):
                    plot_mat[l, shc.lmax + m] = np.log10(np.abs(cqlm[idx, l, m])) if log10_absolute else cqlm[idx, l, m]
                    if m > 0:
                        plot_mat[l, shc.lmax - m] = np.log10(np.abs(sqlm[idx, l, m])) if log10_absolute else sqlm[
                            idx, l, m]

            ax = axes_flat[k]

            degrees, orders = np.arange(0, shc.lmax + 1, 1), np.arange(-shc.lmax, shc.lmax + 1, 1)
            orders2d, degrees2d = np.meshgrid(orders, degrees)
            if shared_colorbar:
                mesh = ax.pcolormesh(
                    orders2d, degrees2d, plot_mat,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    **matshow_kwargs,
                )
            else:
                mesh = ax.pcolormesh(
                    orders2d, degrees2d, plot_mat,
                    cmap=cmap,
                    **matshow_kwargs,
                )
            ax.set_xlabel("order")
            ax.set_ylabel("degree")

            last_mappable = mesh

            if subtitles is not None:
                ax.set_title(str(subtitles[k]), fontsize=10)
            # elif shc.dates_series is not None and idx < len(shc.dates_series):
            #     ax.set_title(str(shc.dates_series[idx]), fontsize=10)
            else:
                ax.set_title(f"Index {idx}", fontsize=10)

            if cbar and not shared_colorbar:
                cb = fig.colorbar(
                    mesh,
                    ax=ax,
                    orientation="vertical",
                    shrink=0.75,
                    pad=0.03,
                )
                if cbar_label is not None:
                    cb.set_label(cbar_label)

        # ----------------------------
        # 7. Hide unused axes
        # ----------------------------
        for k in range(nplot, len(axes_flat)):
            axes_flat[k].set_visible(False)

        # ----------------------------
        # 8. Shared colorbar
        # ----------------------------
        if cbar and shared_colorbar and last_mappable is not None:
            used_axes = axes_flat[:nplot]
            cb = fig.colorbar(
                last_mappable,
                ax=used_axes,
                orientation="vertical",
                shrink=0.85,
                pad=0.03,
            )
            if cbar_label is not None:
                cb.set_label(cbar_label)

        if title is not None:
            fig.suptitle(title, fontsize=13)

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axes

    @shc_plot_method(
        kind="instance",
        summary="2D plot of the spherical harmonic spectrum",
        order=200,
    )
    def triangle(
            self,
            vmin=None,
            vmax=None,
            cbar=True,
            cbar_label=None,
            index=None,
            figsize=None,
            title=None,
            subtitles=None,
            cmap="RdBu_r",
            log10_absolute=False,
            max_plots=24,
            ncols=None,
            ignore_degrees=None,
            savepath=None,
            dpi=300,
            show=True,
            **matshow_kwargs,
    ):
        """
        2D plot of the spherical harmonic spectrum.
        """

        import math
        import matplotlib.pyplot as plt

        shc = self._shc

        # ----------------------------
        # 1. Select indices
        # ----------------------------
        n_total = len(shc)

        if index is None:
            indices = list(range(n_total))
        elif isinstance(index, int):
            indices = [index]
        else:
            indices = list(index)

        if len(indices) == 0:
            raise ValueError("No grid index selected.")

        if len(indices) > max_plots:
            warnings.warn(
                f"Too many grids to plot: {len(indices)}. "
                f"Show only the first {max_plots} indices/grids."
            )
            indices = indices[:max_plots]

        for idx in indices:
            if idx < 0 or idx >= n_total:
                raise IndexError(f"Grid index out of range: {idx}")

        data = shc.value[indices]
        nplot = len(indices)

        # ----------------------------
        # 2. Shared colorbar logic
        # ----------------------------
        shared_colorbar = not (vmin is None and vmax is None)

        if shared_colorbar:
            if vmin is None:
                vmin = np.nanmin(data)
            if vmax is None:
                vmax = np.nanmax(data)

        # ----------------------------
        # 3. Layout
        # ----------------------------
        if ncols is None:
            if nplot == 1:
                ncols = 1
            elif nplot <= 4:
                ncols = 2
            else:
                ncols = 3

        ncols = int(ncols)
        nrows = math.ceil(nplot / ncols)

        if figsize is None:
            figsize = (4.5 * ncols, 3.4 * nrows)

        # ----------------------------
        # 5. Create figure
        # ----------------------------
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            squeeze=False,
        )

        axes_flat = axes.ravel()
        last_mappable = None

        # ----------------------------
        # 6. Plot each panel
        # ----------------------------

        cqlm, sqlm = shc.cs2d

        if ignore_degrees is not None:
            for degree in ignore_degrees:
                cqlm[:, degree, :], sqlm[:, degree, :] = np.nan, np.nan

        for k, idx in enumerate(indices):

            plot_mat = np.full((shc.lmax + 1, 2 * shc.lmax + 1), np.nan)
            for l in range(shc.lmax + 1):
                for m in range(l + 1):
                    plot_mat[l, shc.lmax + m] = np.log10(np.abs(cqlm[idx, l, m])) if log10_absolute else cqlm[idx, l, m]
                    if m > 0:
                        plot_mat[l, shc.lmax - m] = np.log10(np.abs(sqlm[idx, l, m])) if log10_absolute else sqlm[
                            idx, l, m]

            ax = axes_flat[k]

            degrees, orders = np.arange(0, shc.lmax + 1, 1), np.arange(-shc.lmax, shc.lmax + 1, 1)
            orders2d, degrees2d = np.meshgrid(orders, degrees)
            if shared_colorbar:
                mesh = ax.pcolormesh(
                    orders2d, degrees2d, plot_mat,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    **matshow_kwargs,
                )
            else:
                mesh = ax.pcolormesh(
                    orders2d, degrees2d, plot_mat,
                    cmap=cmap,
                    **matshow_kwargs,
                )
            ax.set_xlabel("order")
            ax.set_ylabel("degree")

            last_mappable = mesh

            if subtitles is not None:
                ax.set_title(str(subtitles[k]), fontsize=10)
            # elif shc.dates_series is not None and idx < len(shc.dates_series):
            #     ax.set_title(str(shc.dates_series[idx]), fontsize=10)
            else:
                ax.set_title(f"Index {idx}", fontsize=10)

            if cbar and not shared_colorbar:
                cb = fig.colorbar(
                    mesh,
                    ax=ax,
                    orientation="vertical",
                    shrink=0.75,
                    pad=0.03,
                )
                if cbar_label is not None:
                    cb.set_label(cbar_label)

        # ----------------------------
        # 7. Hide unused axes
        # ----------------------------
        for k in range(nplot, len(axes_flat)):
            axes_flat[k].set_visible(False)

        # ----------------------------
        # 8. Shared colorbar
        # ----------------------------
        if cbar and shared_colorbar and last_mappable is not None:
            used_axes = axes_flat[:nplot]
            cb = fig.colorbar(
                last_mappable,
                ax=used_axes,
                orientation="vertical",
                shrink=0.85,
                pad=0.03,
            )
            if cbar_label is not None:
                cb.set_label(cbar_label)

        if title is not None:
            fig.suptitle(title, fontsize=13)

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axes


class SHCPlotAccessorDescriptor(
    NamespaceAccessorDescriptor[_SHCPlotAccessor]
):
    def __init__(self) -> None:
        super().__init__(_SHCPlotAccessor)
