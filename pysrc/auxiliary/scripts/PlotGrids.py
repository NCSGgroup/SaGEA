"""
This script is to plot spatial distribution through the given 2-d array(s) and latitude and longitude range.
"""
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import matplotlib
from matplotlib import pyplot as plt


def plot_grids(grid: np.ndarray, lat, lon, common_colorbar=False, projection=None, vmin=None, vmax=None, vcenter=None,
               extent=None, subtitle=None, title=None, save=None, cmap=None, cb_extend=None, filling_type=None,
               contour_num=20, gridlines=None, frame_line=None, add_line_lats=None, add_line_lats_color=None,
               add_line_lons=None, add_line_lons_color=None, save_transparent=False):
    """

    :param grid: 2-d array grid or 3-d array grids
    :param lat: array, geophysical latitude in unit[degree]
    :param lon: array, geophysical longitude in unit[degree]
    :param common_colorbar:
    :param projection:
    :param vmin: num or list of num. num for single grid, list of same length with grids for multiple grids
    :param vcenter: num or list of num.
    :param vmax: num or list of num. num for single grid, list of same length with grids for multiple grids
    :param extent: (lonmin, lonmax, latmin, latmax)
    :param subtitle: str or list of str. str for single grid, list of same length with grids for multiple grids
    :param title:
    :param save:
    :param cmap:
    :param cb_extend: 'neither', 'both', 'min', 'max'
    :param filling_type: "gridded", "contour", "contour_filled"
    :param contour_num: int, work only if filling_type in ("contour", "contour_filled")
    :param gridlines:
    :param frame_line:
    :param add_line_lats:
    :param add_line_lons:
    :param save_transparent:
    :return:
    """

    config = {
        "font.family": 'serif',
        "font.size": 16,
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    matplotlib.rcParams.update(config)

    assert grid.ndim in (2, 3)
    assert type(vmin) == type(vmax)
    if filling_type is None:
        filling_type = "gridded"
    assert filling_type in ["gridded", "contour", "contour_filled"]

    if gridlines is None:
        gridlines = (extent is not None)
    if frame_line is None:
        frame_line = True

    if grid.ndim == 2:
        grid = np.array([grid])
    ngrid = np.shape(grid)[0]

    if type(vmin) in (list, tuple,):
        if vcenter is None:
            vcenter = [vcenter] * ngrid
    else:
        vmin = [vmin] * ngrid
        vmax = [vmax] * ngrid
        vcenter = [vcenter] * ngrid

    if subtitle is None:
        subtitle = ''

    if type(subtitle) in (list, tuple,):
        pass
    else:
        subtitle = [subtitle] * ngrid

    if grid.ndim == 3:
        assert len(grid) == len(subtitle) == len(vmin) == len(vmax)

    if ngrid == 1:
        assert not common_colorbar, "not support yet"

        fig = plt.figure(figsize=(6, 4))

        axes_loc = (
            (0.05, 0.3),  # plot grids

            (0.05, 0.015),  # plot color bars

            (0.025, -0.025),  # plot subtitles

            (0.3, 0.8),  # plot title
        )

        axes_size = (
            (0.9, 0.6),  # plot grids

            (0.9, 0.2),  # plot color bars

            (0.95, 0.2),  # plot subtitles

            (0.4, 0.1),  # plot title
        )

    elif ngrid == 3:
        assert not common_colorbar, "not support yet"

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

    elif ngrid == 4:
        fig = plt.figure(figsize=(9, 6))

        if common_colorbar:
            axes_loc = (
                (0.025, 0.5 + 0.1),  # plot grids
                (0.5 + 0.025, 0.5 + 0.1),
                (0.025, 0.175),
                (0.5 + 0.025, 0.175),

                (0.1, 0.05),  # plot colorbar

                (0., 0.5 - 0.025),  # plot subtitles
                (0.5 + 0., 0.5 - 0.025),
                (0., 0.05),
                (0.5 + 0., 0.05),

                (0.3, 0.85),  # plot title
            )

            axes_size = (
                (0.45, 0.35),  # plot grids
                (0.45, 0.35),
                (0.45, 0.35),
                (0.45, 0.35),

                (0.8, 0.05),  # plot colorbar

                (0.5, 0.2),  # plot subtitles
                (0.5, 0.2),
                (0.5, 0.2),
                (0.5, 0.2),

                (0.4, 0.1),  # plot title
            )

        else:
            axes_loc = (
                (0.025, 0.5 + 0.1),  # plot grids
                (0.5 + 0.025, 0.5 + 0.1),  # plot grids
                (0.025, 0.125),
                (0.5 + 0.025, 0.125),

                (0.025, 0.5 - 0.125),  # plot colorbars
                (0.5 + 0.025, 0.5 - 0.125),
                (0.025, 0. - 0.1),
                (0.5 + 0.025, 0. - 0.1),

                (0., 0.5 - 0.1),  # plot subtitles
                (0.5 + 0., 0.5 - 0.1),
                (0., 0. - 0.075),
                (0.5 + 0., 0. - 0.075),

                (0.3, 0.85),  # plot title
            )

            axes_size = (
                (0.45, 0.35),  # plot grids
                (0.45, 0.35),
                (0.45, 0.35),
                (0.45, 0.35),

                (0.45, 0.2),  # plot colorbars
                (0.45, 0.2),
                (0.45, 0.2),
                (0.45, 0.2),

                (0.5, 0.2),  # plot subtitles
                (0.5, 0.2),
                (0.5, 0.2),
                (0.5, 0.2),

                (0.4, 0.1),  # plot title
            )

    else:
        return -1

    if projection is None:
        projection = ccrs.PlateCarree()

    axes_grids = [
        fig.add_axes([*axes_loc[i], *axes_size[i]], projection=projection)
        for i in range(ngrid)]

    if common_colorbar:
        axes_cbs = [fig.add_axes([*axes_loc[i + ngrid], *axes_size[i + ngrid]]) for i in range(1)]
        axes_subtitles = [fig.add_axes([*axes_loc[i + 1 + ngrid], *axes_size[i + 1 + ngrid]]) for i in range(ngrid)]

    else:
        axes_cbs = [fig.add_axes([*axes_loc[i + ngrid], *axes_size[i + ngrid]]) for i in range(ngrid)]
        axes_subtitles = [fig.add_axes([*axes_loc[i + 2 * ngrid], *axes_size[i + 2 * ngrid]]) for i in range(ngrid)]

    ax_title = fig.add_axes([*axes_loc[-1], *axes_size[-1]])

    lon2d, lat2d = np.meshgrid(lon, lat)
    for i in range(ngrid):
        ax_grid = axes_grids[i]

        if common_colorbar:
            ax_cb = axes_cbs[0]
        else:
            ax_cb = axes_cbs[i]

        ax_subtitle = axes_subtitles[i]

        if vmin[i] is None or vmax[i] is None:
            assert vcenter[i] is None
            norm = None
        else:
            this_vcenter = ((vmax[i] + vmin[i]) / 2) if vcenter[i] is None else vcenter[i]
            norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin[i], vmax=vmax[i], vcenter=this_vcenter)

        if filling_type == "gridded":
            p = ax_grid.pcolormesh(
                lon2d, lat2d, grid[i],
                # cmap=cmaps.matlab_jet,
                cmap=cmap if cmap is not None else cmaps.matlab_jet,
                transform=ccrs.PlateCarree(),
                norm=norm
            )

        else:
            p = ax_grid.pcolormesh(
                lon2d[:1, :1], lat2d[:1, :1], grid[i][:1, :1],
                cmap=cmap if cmap is not None else cmaps.matlab_jet,
                transform=ccrs.PlateCarree(),
                norm=norm
            )

            if filling_type == "contour":
                ax_grid.contour(
                    lon2d, lat2d, grid[i], contour_num,
                    cmap=cmap if cmap is not None else cmaps.matlab_jet,
                    transform=ccrs.PlateCarree(),
                    norm=norm
                )

            elif filling_type == "contour_filled":
                ax_grid.contourf(
                    lon2d, lat2d, grid[i], contour_num,
                    cmap=cmap if cmap is not None else cmaps.matlab_jet,
                    transform=ccrs.PlateCarree(),
                    norm=norm
                )

            else:
                assert False

        if gridlines:
            ax_grid.gridlines()

        if extent is not None:
            ax_grid.set_extent(extent)
            ax_grid.set_xticks(np.linspace(extent[0], extent[1], 3), crs=ccrs.PlateCarree(), minor=True)
            ax_grid.set_yticks(np.linspace(extent[2], extent[3], 3), crs=ccrs.PlateCarree(), minor=True)

        if not frame_line:
            ax_grid.axis("off")

        '''experiment'''
        if add_line_lats is not None:
            for la in range(len(add_line_lats)):
                ax_grid.plot([-180, 180], [add_line_lats[la]] * 2, c=add_line_lats_color[la], lw=1.5,
                             transform=ccrs.PlateCarree())

        if add_line_lons is not None:
            for lo in range(len(add_line_lons)):
                ax_grid.plot([add_line_lons[lo]] * 2, [-90, 90], c=add_line_lons_color[lo], lw=1.5,
                             transform=ccrs.PlateCarree())

        ax_grid.add_feature(cfeature.COASTLINE)

        ax_subtitle.axis('off')
        ax_subtitle.set_xlim(-1, 1)
        ax_subtitle.set_ylim(-1, 1)
        ax_subtitle.text(
            0, 0, subtitle[i],
            verticalalignment='center',
            horizontalalignment='center'
        )

        if common_colorbar and i > 0:
            continue

        ax_cb.axis('off')
        ax_cb.set_xlim(-1, 1)
        ax_cb.set_ylim(-1, 1)
        cb = fig.colorbar(p,
                          orientation='horizontal',
                          fraction=1, ax=ax_cb,
                          aspect=30,
                          extend=cb_extend if cb_extend is not None else 'both',
                          )
        # cb.mappable.set_clim(-0.2, 1.0)
        cb.ax.tick_params(direction='in')

    ax_title.axis('off')
    ax_title.set_xlim(-1, 1)
    ax_title.set_ylim(-1, 1)
    if title is not None:
        ax_title.set_title(title)
    if save is not None:
        plt.savefig(save, transparent=save_transparent, dpi=200)

    plt.show()
    plt.close()
