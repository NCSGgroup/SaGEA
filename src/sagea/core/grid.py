#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/6/17 12:45 
# @File    : grid.py

from __future__ import annotations

import datetime

import numpy as np

from sagea.harmonics import Harmonic, GRDType
from sagea.utils.MathTool import MathTool


class GRD:
    """
    Gridded data container.

    Parameters
    ----------
    grid : ndarray
        Shape:
        - (nlat, nlon)
        - (ntime, nlat, nlon)

    lat : ndarray
        Latitude in degrees by default.
        If option=0, input is colatitude in radians.

    lon : ndarray
        Longitude in degrees by default.
        If option=0, input is longitude in radians.

    option : int
        - 0: input lat is colatitude in radians, lon is radians
        - 1: input lat/lon are degrees
    """

    def __init__(
            self,
            grid: np.ndarray,
            lat: np.ndarray,
            lon: np.ndarray,
            option: int = 1,
    ):
        grid = np.asarray(grid, dtype=float)

        if grid.ndim == 2:
            grid = grid[None, :, :]

        if grid.ndim != 3:
            raise ValueError("grid should be a 2D or 3D array.")

        if grid.shape[-2:] != (len(lat), len(lon)):
            raise ValueError(
                f"grid shape mismatch. Expected (..., {len(lat)}, {len(lon)}), "
                f"got {grid.shape}."
            )

        self.value = grid

        if option == 0:
            self.lat = 90 - np.degrees(lat)
            self.lon = np.degrees(lon)
        elif option == 1:
            self.lat = np.asarray(lat, dtype=float)
            self.lon = np.asarray(lon, dtype=float)
        else:
            raise ValueError("option should be 0 or 1.")

        self.__grid_type: GRDType | None = None
        self.dates_series: list[datetime.date] | None = None

    def __len__(self) -> int:
        return self.value.shape[0]

    @property
    def shape(self):
        return self.value.shape

    @property
    def grid_type(self) -> GRDType | None:
        return self.__grid_type

    @grid_type.setter
    def grid_type(self, grid_type: GRDType | None):
        if grid_type is not None and not isinstance(grid_type, GRDType):
            raise TypeError("grid_type should be GRDType or None.")
        self.__grid_type = grid_type

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.value, axis=0)

    @property
    def std(self) -> np.ndarray:
        return np.std(self.value, axis=0)

    def copy(self) -> "GRD":
        new = GRD(self.value.copy(), self.lat.copy(), self.lon.copy())
        new.grid_type = self.grid_type
        new.dates_series = None if self.dates_series is None else list(self.dates_series)
        return new

    def limiter(
            self,
            threshold: float = 0.5,
            beyond: float = 1,
            below: float = 0,
    ) -> "GRD":
        self.value = np.where(self.value >= threshold, beyond, below)
        return self

    # def de_aliasing(
    #     self,
    #     dates: list[datetime.date],
    #     tide_periods: dict[str, float],
    # ) -> "GRD":
    #     """
    #     Remove tidal aliasing signals from grid time series.
    #     """
    #
    #     from sagea.processing.DeAliasing import grid_tide_de_aliasing
    #
    #     self.value = grid_tide_de_aliasing(self.value, dates, tide_periods)
    #     return self
    #
    # def regional_extraction(
    #     self,
    #     mask: np.ndarray,
    #     average: bool = True,
    #     leakage: constant.LeakageMethod | None = None,
    #     **kwargs: Any,
    # ) -> np.ndarray:
    #     """
    #     Extract regional average or total signal.
    #
    #     Parameters
    #     ----------
    #     mask : ndarray
    #         Region mask with shape (nlat, nlon).
    #
    #     average : bool
    #         If True, return regional average.
    #         If False, return regional integral.
    #
    #     leakage : LeakageMethod or None
    #         Leakage correction method.
    #     """
    #
    #     from sagea.processing.leakage.Additive import additive
    #     from sagea.processing.leakage.Multiplicative import multiplicative
    #     from sagea.processing.leakage.Scaling import scaling
    #     from sagea.processing.leakage.ScalingGrid import scaling_grid
    #     from sagea.processing.leakage.ForwardModeling import forward_modeling
    #     from sagea.processing.leakage.DataDriven import data_driven
    #     from sagea.processing.leakage.BufferZone import buffer_zone
    #
    #     mask = np.asarray(mask)
    #
    #     if mask.shape != self.value.shape[-2:]:
    #         raise ValueError(
    #             f"mask shape mismatch. Expected {self.value.shape[-2:]}, got {mask.shape}."
    #         )
    #
    #     dispatch = {
    #         None: None,
    #         constant.LeakageMethod.Additive: additive,
    #         constant.LeakageMethod.Multiplicative: multiplicative,
    #         constant.LeakageMethod.Scaling: scaling,
    #         constant.LeakageMethod.ScalingGrid: scaling_grid,
    #         constant.LeakageMethod.DataDriven: data_driven,
    #         constant.LeakageMethod.ForwardModeling: forward_modeling,
    #         constant.LeakageMethod.BufferZone: buffer_zone,
    #     }
    #
    #     if leakage not in dispatch:
    #         raise ValueError(f"Unsupported leakage method: {leakage}")
    #
    #     leakage_corrector = dispatch[leakage]
    #
    #     if leakage_corrector is not None:
    #         if not average:
    #             raise ValueError("average must be True when leakage correction is used.")
    #
    #         return leakage_corrector(
    #             grid_value=self.value,
    #             lat=self.lat,
    #             lon=self.lon,
    #             basin_mask=mask,
    #             **kwargs,
    #         )
    #
    #     result = MathTool.global_integral(self.value * mask, self.lat, self.lon)
    #
    #     if average:
    #         result = result / MathTool.get_acreage(mask)
    #
    #     return result

    def to_SHC(self, lmax: int):
        """
        Pure harmonic analysis.

        Returns
        -------
        SHC
        """

        from sagea.core.shc import SHC

        lmax = int(lmax)

        if self.grid_type is None:
            har = Harmonic(
                lmax=lmax,
                grid_type=None,
                lat=self.lat,
                lon=self.lon,
            )
        else:
            har = Harmonic(
                lmax=lmax,
                grid_type=self.grid_type,
            )

        cqlm, sqlm = har.analysis(self.value)
        cs = MathTool.cs_combine_to_triangle_1d(cqlm, sqlm)

        return SHC(cs)

    def plot(
            self,
            vmin=None,
            vmax=None,
            projection=None,
            index=None,
            cmap="RdBu_r",
            figsize=None,
            title=None,
            titles=None,
            cbar_label=None,
            cbar=True,
            coastline=True,
            borders=False,
            gridlines=True,
            extent=None,
            add_cyclic=True,
            max_plots=24,
            ncols=None,
            shading="auto",
            transform=None,
            savepath=None,
            dpi=300,
            show=True,
            **pcolormesh_kwargs,
    ):
        """
        Plot gridded data on maps.

        Parameters
        ----------
        vmin, vmax : float or None
            Color scale range.
            If both are None, each subplot uses its own colorbar.
            If either vmin or vmax is given, all subplots share one colorbar.

        projection : cartopy.crs projection or None
            Map projection. Default is cartopy.crs.PlateCarree().

        index : int, sequence of int, or None
            Indices of grids to plot.
            If None, plot all grids, but total number must be <= max_plots.

        cmap : str or Colormap
            Matplotlib colormap.

        figsize : tuple or None
            Figure size. If None, automatically determined.

        title : str or None
            Figure suptitle.

        titles : sequence of str or None
            Titles for each subplot. Length should match selected indices.

        cbar_label : str or None
            Colorbar label.

        cbar : bool
            Whether to draw colorbar.

        coastline : bool
            Whether to draw coastlines.

        borders : bool
            Whether to draw national borders.

        gridlines : bool
            Whether to draw gridlines.

        extent : tuple or None
            Map extent in degrees: (lon_min, lon_max, lat_min, lat_max).

        add_cyclic : bool
            Whether to add cyclic point in longitude to avoid map seam.

        max_plots : int
            Maximum number of subplots allowed.

        ncols : int or None
            Number of columns. If None, automatically determined.
            Default automatic strategy keeps at most 3 columns, so 24 panels -> 8x3.

        shading : str
            Passed to matplotlib/cartopy pcolormesh.

        transform : cartopy.crs projection or None
            Coordinate CRS of input data. Default is PlateCarree.

        savepath : str or pathlib.Path or None
            If given, save figure to this path.

        dpi : int
            DPI for saving.

        show : bool
            Whether to call plt.show().

        **pcolormesh_kwargs
            Extra keyword arguments passed to ax.pcolormesh().

        Returns
        -------
        fig, axes
            Matplotlib figure and axes array.
        """

        import math
        import numpy as np
        import matplotlib.pyplot as plt

        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            from cartopy.util import add_cyclic_point
        except ImportError as exc:
            raise ImportError(
                "GRD.plot() requires cartopy. Install it with:\n"
                "    pip install cartopy"
            ) from exc

        # ----------------------------
        # 1. Projection and transform
        # ----------------------------
        if projection is None:
            projection = ccrs.PlateCarree()

        if transform is None:
            transform = ccrs.PlateCarree()

        # ----------------------------
        # 2. Select indices
        # ----------------------------
        n_total = len(self)

        if index is None:
            indices = list(range(n_total))
        elif isinstance(index, int):
            indices = [index]
        else:
            indices = list(index)

        if len(indices) == 0:
            raise ValueError("No grid index selected.")

        if len(indices) > max_plots:
            raise ValueError(
                f"Too many grids to plot: {len(indices)}. "
                f"Please specify index or increase max_plots."
            )

        for idx in indices:
            if idx < 0 or idx >= n_total:
                raise IndexError(f"Grid index out of range: {idx}")

        data = self.value[indices]
        nplot = len(indices)

        # ----------------------------
        # 3. Shared colorbar logic
        # ----------------------------
        shared_colorbar = not (vmin is None and vmax is None)

        if shared_colorbar:
            if vmin is None:
                vmin = np.nanmin(data)
            if vmax is None:
                vmax = np.nanmax(data)

        # ----------------------------
        # 4. Layout
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
            subplot_kw={"projection": projection},
            squeeze=False,
        )

        axes_flat = axes.ravel()

        lon = np.asarray(self.lon)
        lat = np.asarray(self.lat)

        last_mappable = None

        # ----------------------------
        # 6. Plot each panel
        # ----------------------------
        for k, idx in enumerate(indices):
            ax = axes_flat[k]
            z = self.value[idx]

            plot_lon = lon
            plot_z = z

            if add_cyclic:
                try:
                    plot_z, plot_lon = add_cyclic_point(z, coord=lon)
                except Exception:
                    plot_z, plot_lon = z, lon

            if shared_colorbar:
                mesh = ax.pcolormesh(
                    plot_lon,
                    lat,
                    plot_z,
                    transform=transform,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    shading=shading,
                    **pcolormesh_kwargs,
                )
            else:
                mesh = ax.pcolormesh(
                    plot_lon,
                    lat,
                    plot_z,
                    transform=transform,
                    cmap=cmap,
                    shading=shading,
                    **pcolormesh_kwargs,
                )

            last_mappable = mesh

            if coastline:
                ax.coastlines(linewidth=0.6)

            if borders:
                ax.add_feature(cfeature.BORDERS, linewidth=0.4)

            if extent is not None:
                ax.set_extent(extent, crs=ccrs.PlateCarree())

            if gridlines:
                gl = ax.gridlines(
                    draw_labels=True,
                    linewidth=0.3,
                    color="gray",
                    alpha=0.5,
                    linestyle="--",
                )
                gl.top_labels = False
                gl.right_labels = False

            if titles is not None:
                ax.set_title(str(titles[k]), fontsize=10)
            elif self.dates_series is not None and idx < len(self.dates_series):
                ax.set_title(str(self.dates_series[idx]), fontsize=10)
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

        # fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

        if show:
            plt.show()

        return fig, axes
