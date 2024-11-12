import matplotlib.pyplot as plt
import numpy as np
import datetime

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH, load_SHC
import pysrc.auxiliary.preference.EnumClasses as Enums
from pysrc.auxiliary.scripts.PlotGrids import plot_grids

from pysrc.uncertainty_estimating.three_coener_hat.TCH import TCHMode, tch_estimate


def demo1():
    """this demo shows how to get variance of post-processed GRACE data by TCH: a global distribution"""

    '''TCH config'''
    tch_mode = TCHMode.OLS

    '''loading files and post-processing'''
    gif48_path = FileTool.get_project_dir() / 'data/auxiliary/GIF48.gfc'
    shc_bg = load_SHC(gif48_path, key='gfc', lmax=60)

    load = LoadL2SH()
    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2015, 12, 31))

    load.configuration.set_institute(Enums.L2InstituteType.CSR)
    shc_csr = load.get_shc()
    shc_csr.de_background(shc_bg)

    load.configuration.set_institute(Enums.L2InstituteType.GFZ)
    shc_gfz = load.get_shc()
    shc_gfz.de_background(shc_bg)

    load.configuration.set_institute(Enums.L2InstituteType.JPL)
    shc_jpl = load.get_shc()
    shc_jpl.de_background(shc_bg)

    shc_csr.value[:, :6] = 0
    shc_gfz.value[:, :6] = 0
    shc_jpl.value[:, :6] = 0

    shcs = [shc_csr, shc_gfz, shc_jpl]

    '''filter'''
    for i in range(len(shcs)):
        shcs[i].filter(method=Enums.SHCFilterType.Gaussian, param=(300,))

    '''convert to ewh'''
    for i in range(len(shcs)):
        shcs[i].convert_type(from_type=Enums.PhysicalDimensions.Dimensionless, to_type=Enums.PhysicalDimensions.EWH)

    '''harmonic synthesis to gridded signal'''
    grid_space = 1
    grids = []
    for i in range(len(shcs)):
        grids.append(shcs[i].to_grid(grid_space=grid_space))

    '''tch for grid'''
    grids_value = [grids[i].value for i in range(len(grids))]
    variance = tch_estimate(*grids_value, mode=tch_mode)

    '''plot'''
    plot_grids(
        np.sqrt(variance) * 100,  # cm
        lat=grids[0].lat,
        lon=grids[1].lon,
        vmin=0.,
        vmax=6.,
        subtitle=['CSR', 'GFZ', 'JPL'],
        title=f"TCH with {tch_mode.name}"
    )


def demo2():
    """this demo shows how to get variance of post-processed GRACE data by TCH: CS distribution"""

    '''TCH config'''
    tch_mode = TCHMode.OLS

    '''loading files and post-processing'''
    gif48_path = FileTool.get_project_dir() / 'data/auxiliary/GIF48.gfc'
    shc_bg = load_SHC(gif48_path, key='gfc', lmax=60)

    load = LoadL2SH()
    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2015, 12, 31))

    load.configuration.set_institute(Enums.L2InstituteType.CSR)
    shc_csr = load.get_shc()
    shc_csr.de_background(shc_bg)

    load.configuration.set_institute(Enums.L2InstituteType.GFZ)
    shc_gfz = load.get_shc()
    shc_gfz.de_background(shc_bg)

    load.configuration.set_institute(Enums.L2InstituteType.JPL)
    shc_jpl = load.get_shc()
    shc_jpl.de_background(shc_bg)

    shc_csr.value[:, :6] = 0
    shc_gfz.value[:, :6] = 0
    shc_jpl.value[:, :6] = 0

    shcs = [shc_csr, shc_gfz, shc_jpl]

    for i in range(len(shcs)):
        shcs[i].filter(method=Enums.SHCFilterType.Gaussian, param=(300,))

    '''tch for CS'''
    cs_value = [shcs[i].value for i in range(len(shcs))]
    variance = tch_estimate(*cs_value, mode=tch_mode)

    cs_matrices = []
    for i in range(len(variance)):
        c, s = MathTool.cs_decompose_triangle1d_to_cs2d(np.sqrt(variance[i]), fill=np.nan)
        cs = MathTool.cs_combine_to_triangle(c, s)
        cs_matrices.append(cs)

    '''plot'''
    subtitles = ("CSR", "GFZ", "JPL")

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax_cb = fig.add_axes([0.25, 0.05, 0.5, 0.2])

    axes = [ax1, ax2, ax3]
    p = None
    for i in range(len(axes)):
        ax = axes[i]
        p = ax.matshow(np.sqrt(cs_matrices[i]), vmin=0, vmax=1e-6)

        ax.set_xticks([0, 30, 60, 90, 120], [-60, -30, 0, 30, 60])
        ax.set_yticks([0, 30, 60])

        ax.set_xlabel("degree")
        if i == 0:
            ax.set_ylabel("order")

        ax.set_title(subtitles[i])

    ax_cb.axis('off')
    ax_cb.set_xlim(-1, 1)
    ax_cb.set_ylim(-1, 1)
    cb = fig.colorbar(p,
                      orientation='horizontal',
                      fraction=1, ax=ax_cb,
                      aspect=30,
                      extend='both',
                      )
    cb.ax.tick_params(direction='in')

    plt.show()


if __name__ == '__main__':
    demo1()
