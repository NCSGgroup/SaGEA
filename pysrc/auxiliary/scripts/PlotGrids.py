"""
This script is to plot spatial distribution through the given 2-d array(s) and latitude and longitude range.
"""
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import matplotlib
from matplotlib import pyplot as plt


def plot_grids(grid: np.ndarray, lat, lon, vmin, vmax, central_longitude=None,
               extent=None, subtitle=None, title=None, save=None):
    """

    :param grid: 2-d array grid or 3-d array grids
    :param lat: array, geophysical latitude in unit[degree]
    :param lon: array, geophysical longitude in unit[degree]
    :param vmin: num or list of num. num for single grid, list of same length with grids for multiple grids
    :param vmax: num or list of num. num for single grid, list of same length with grids for multiple grids
    :param central_longitude:
    :param extent:
    :param subtitle: str or list of str. str for single grid, list of same length with grids for multiple grids
    :param title: str or list of str. str for single grid, list of same length with grids for multiple grids
    :param save: str or list of str. str for single grid, list of same length with grids for multiple grids
    :param save: str or list of str. str for single grid, list of same length with grids for multiple grids
    :return:
    """
    assert grid.ndim in (2, 3)
    assert type(vmin) == type(vmax)

    if grid.ndim == 2:
        grid = np.array([grid])
    ngrid = np.shape(grid)[0]

    if type(vmin) in (list,):
        pass
    else:
        vmin = [vmin] * ngrid
        vmax = [vmax] * ngrid

    if subtitle is None:
        subtitle = ''

    if type(subtitle) in (list,):
        pass
    else:
        subtitle = [subtitle] * ngrid

    if central_longitude is None:
        central_longitude = 0

    if grid.ndim == 3:
        assert len(grid) == len(subtitle) == len(vmin) == len(vmax)

    if ngrid == 1:
        fig = plt.figure(figsize=(6, 4))

        axes_loc = (
            (0.05, 0.3),  # plot grids

            (0.05, 0.005),  # plot color bars

            (0.025, 0.),  # plot subtitles

            (0.3, 0.8),  # plot title
        )

        axes_size = (
            (0.9, 0.6),  # plot grids

            (0.9, 0.2),  # plot color bars

            (0.95, 0.2),  # plot subtitles

            (0.4, 0.1),  # plot title
        )

    elif ngrid == 3:
        fig = plt.figure(figsize=(10, 3))

        axes_loc = (
            (0.025, 0.3),  # plot grids
            (0.025 + (0.3 + 0.025), 0.3),
            (0.025 + (0.3 + 0.025) * 2, 0.3),

            (0.025, 0.1),  # plot colorbars
            (0.025 + (0.3 + 0.025), 0.1),
            (0.025 + (0.3 + 0.025) * 2, 0.1),

            (0.025, 0.),  # plot subtitles
            (0.025 + (0.3 + 0.025), 0.),
            (0.025 + (0.3 + 0.025) * 2, 0.),

            (0.3, 0.8),  # plot title
        )

        axes_size = (
            (0.3, 0.6),  # plot grids
            (0.3, 0.6),
            (0.3, 0.6),

            (0.3, 0.2),  # plot colorbars
            (0.3, 0.2),
            (0.3, 0.2),

            (0.3, 0.2),  # plot subtitles
            (0.3, 0.2),
            (0.3, 0.2),

            (0.4, 0.1),  # plot title
        )

    else:
        return -1

    axes_grids = [
        fig.add_axes([*axes_loc[i], *axes_size[i]], projection=ccrs.PlateCarree(central_longitude=central_longitude))
        for i in range(ngrid)]
    axes_cbs = [fig.add_axes([*axes_loc[i + ngrid], *axes_size[i + ngrid]]) for i in range(ngrid)]
    axes_subtitles = [fig.add_axes([*axes_loc[i + 2 * ngrid], *axes_size[i + 2 * ngrid]]) for i in range(ngrid)]
    ax_title = fig.add_axes([*axes_loc[-1], *axes_size[-1]])

    lon2d, lat2d = np.meshgrid(lon, lat)
    for i in range(ngrid):
        ax_grid = axes_grids[i]
        ax_cb = axes_cbs[i]
        ax_subtitle = axes_subtitles[i]

        if vmin[i] is None or vmax[i] is None:
            norm = None
            # vcenter = None
        else:
            vcenter = (vmax[i] + vmin[i]) / 2
            norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin[i], vmax=vmax[i], vcenter=vcenter)

        p = ax_grid.pcolormesh(
            lon2d, lat2d, grid[i],
            cmap=cmaps.matlab_jet,
            transform=ccrs.PlateCarree(),
            norm=norm
        )

        if extent is not None:
            ax_grid.set_extent(extent)
            ax_grid.gridlines()
            ax_grid.set_xticks(np.linspace(extent[0], extent[1], 3), crs=ccrs.PlateCarree(), minor=True)
            ax_grid.set_yticks(np.linspace(extent[2], extent[3], 3), crs=ccrs.PlateCarree(), minor=True)

        ax_grid.add_feature(cfeature.COASTLINE)

        ax_cb.axis('off')
        ax_cb.set_xlim(-1, 1)
        ax_cb.set_ylim(-1, 1)
        cb = fig.colorbar(p,
                          orientation='horizontal',
                          fraction=1, ax=ax_cb,
                          #  ticks=np.linspace(vmin[i], vmax[i], 3)
                          aspect=30,
                          extend='both',
                          )
        cb.ax.tick_params(direction='in')

        ax_subtitle.axis('off')
        ax_subtitle.set_xlim(-1, 1)
        ax_subtitle.set_ylim(-1, 1)
        ax_subtitle.text(
            0, 0, subtitle[i],
            verticalalignment='center',
            horizontalalignment='center'
        )

    ax_title.axis('off')
    ax_title.set_xlim(-1, 1)
    ax_title.set_ylim(-1, 1)
    if title is not None:
        ax_title.set_title(title)
    if save is not None:
        plt.savefig(save)

    plt.show()
    plt.close()

    # break


def demo():
    from pysrc.auxiliary.aux_tool.MathTool import MathTool
    grid1 = np.load('../../../results/spatial_std/2009-06_1.npy') * 1000

    grid2 = np.load('../../../results/spatial_std/sigmaEWH_200906_ITSG_lmax60_diag_GS300.npy') * 1000

    grid3 = grid1 - grid2

    lat, lon = MathTool.get_global_lat_lon_range(1)
    plot_grids(
        grid=np.array([grid1, grid2, grid3]), lat=lat, lon=lon,
        vmin=[0, 0, -1],
        vmax=[30, 30, 1],
        subtitle=('(a) Yang', '(b) Liu', '(a)-(b)'),
        title='sigma EWH 200906 diag matrix GS300',
        save='../../../results/spatial_std/compare_sigmaEWH_200906_diag.pdf'
    )


if __name__ == '__main__':
    demo()
