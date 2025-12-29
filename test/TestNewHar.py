import pathlib

import cartopy.crs
import cmaps
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.auxiliary.preference.EnumClasses import PhysicalDimensions, SHCFilterType
from pysrc.auxiliary.scripts.PlotGrids import plot_grids
from pysrc.post_processing.harmonic.Harmonic import GRDType


def plt_config():
    config = {
        "font.family": 'serif',
        "font.serif": ["STFangsong"],
        "font.size": 12,
        "axes.unicode_minus": False,
        "mathtext.fontset": 'stix',
    }

    matplotlib.rcParams.update(config)


def demo():
    # plt_config()

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    gsm_paths = list(pathlib.Path(
        r"../data/L2_SH_products/GSM/CSR/RL06/BA01/2005"
    ).iterdir())
    lmax = 60
    grid_type = GRDType.GLQ
    grid_space = 1

    shc = load_SHC(*gsm_paths, key="GRCOF2", lmax=lmax)
    shc.convert_type(from_type=PhysicalDimensions.Dimensionless, to_type=PhysicalDimensions.EWH)
    shc.de_background()
    shc.filter(SHCFilterType.Gaussian, (300,))

    # grd = shc.to_GRD(grid_type=grid_type)
    # shc2 = grd.to_SHC()
    # grd2 = shc2.to_GRD(grid_type=grid_type)

    grd_simp = shc.to_GRD(3)
    shc2_simp = grd_simp.to_SHC(60)
    grd2_simp = shc2_simp.to_GRD(3)

    grd = shc.to_GRD(grid_type=grid_type)
    shc2 = grd.to_SHC()
    grd2 = shc2.to_GRD(grid_type=grid_type)

    print(grd2_simp.value.shape, grd2.value.shape)

    fig = plt.figure(figsize=(8, 2.5))

    ax1 = fig.add_axes([0.020, 0.30, 0.3, 0.8], projection=cartopy.crs.PlateCarree())
    ax2 = fig.add_axes([0.350, 0.30, 0.3, 0.8], projection=cartopy.crs.PlateCarree())
    ax3 = fig.add_axes([0.680, 0.30, 0.3, 0.8], projection=cartopy.crs.PlateCarree())

    ax1_cb = fig.add_axes([0.020, 0.30, 0.3, 0.1])
    ax2_cb = fig.add_axes([0.350, 0.30, 0.3, 0.1])
    ax3_cb = fig.add_axes([0.680, 0.30, 0.3, 0.1])

    grid_to_plot = (grd, grd2_simp - grd_simp, grd2 - grd)
    ax_to_plot = (ax1, ax2, ax3)
    axes_cb = (ax1_cb, ax2_cb, ax3_cb)
    vmin = (-15, -2e-1, -4e-14)
    vmax = (15, 2e-1, 4e-14)
    cmap = ("jet", "RdBu_r", "RdBu_r")

    # ax_frame = fig.add_axes((0, 0, 1, 1))
    # ax_frame.axis("off")
    # ax_frame.text(0.17, 0.15, r"$\mathrm{(a)}$ 原始信号", ha="center", va="center")

    for i in range(len(ax_to_plot)):
        grid = grid_to_plot[i]
        ax = ax_to_plot[i]
        ax_cb = axes_cb[i]

        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin[i], vmax=vmax[i], vcenter=0)

        lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)

        p = ax.pcolormesh(
            lon2d[:1, :1], lat2d[:1, :1], grid.value[0][:1, :1] * 100,
            transform=cartopy.crs.PlateCarree(),
            norm=norm,
            cmap=cmap[i],
            zorder=-1
        )

        ax.contourf(
            lon2d, lat2d, grid.value[0] * 100, 30,
            transform=cartopy.crs.PlateCarree(),
            norm=norm,
            cmap=cmap[i]
        )

        ax.add_feature(cartopy.feature.COASTLINE)

        ax_cb.axis('off')
        ax_cb.set_xlim(-1, 1)
        ax_cb.set_ylim(-1, 1)
        cb = fig.colorbar(p,
                          orientation='horizontal',
                          fraction=1, ax=ax_cb,
                          aspect=30,
                          extend='both',
                          )
        # cb.mappable.set_clim(-0.2, 1.0)
        cb.ax.tick_params(direction='in')

    fig.savefig("球谐分析权重因子.png", dpi=220)
    fig.show()


if __name__ == '__main__':
    demo()
