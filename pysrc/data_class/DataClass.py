import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool

from pysrc.post_processing.harmonic.Harmonic import Harmonic, CoreSHC, CoreGRID


class SHC(CoreSHC):
    def __init__(self, c, s=None):
        super().__init__(c, s)

    def to_grid(self, grid_space=None):
        if grid_space is None:
            grid_space = int(180 / self.get_lmax())

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        lmax = self.get_lmax()
        har = Harmonic(lat, lon, lmax, option=1)

        cqlm, sqlm = self.get_cs2d()
        grid_data = har.synthesis_for_csqlm(cqlm, sqlm)
        grid = GRID(grid_data, lat, lon, option=1)

        return grid


class GRID(CoreGRID):
    def __init__(self, grid, lat, lon, option=0):
        super().__init__(grid, lat, lon, option)

    def to_SHC(self, lmax=None):
        grid_space = self.get_grid_space()

        if lmax is None:
            lmax = int(180 / grid_space)

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        har = Harmonic(lat, lon, lmax, option=1)

        grid_data = self.data
        cqlm, sqlm = har.analysis_for_gqij(grid_data)
        shc = SHC(cqlm, sqlm)

        return shc


def demo():
    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    lmax = 60
    grid_space = 1

    c, s = load_SH_simple(
        FileTool.get_project_dir(
            'data/L2_SH_products/GSM/CSR/RL06/BA01/2002/GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600'),
        key='GRCOF2',
        lmax=lmax
    )
    c[0, 0] = 0

    shc = SHC(c, s)

    grid = shc.to_grid(grid_space)

    shc_new = grid.to_SHC(lmax=lmax)

    diff_cs1d = np.abs((shc.cs[0] - shc_new.cs[0]))
    # diff_cs1d = shc_new.cs[0]

    # begin_order = 3
    # cl0_even = MathTool.cs_decompose_triangle1d_to_cs2d(diff_cs1d)[0][begin_order:, begin_order]
    # plt.plot(cl0_even)
    # # plt.ylim(0.955, 1.035)
    # plt.show()
    # print(np.sum(diff_cs1d ** 2))

    diff_cs_tri = MathTool.cs_combine_to_triangle(*MathTool.cs_decompose_triangle1d_to_cs2d(diff_cs1d))

    for i in range(len(diff_cs_tri)):
        for j in range(len(diff_cs_tri[i])):
            if i < np.abs(j - lmax):
                diff_cs_tri[i, j] = np.nan

    plt.matshow(diff_cs_tri)
    # plt.matshow(diff_cs_tri, vmin=0, vmax=1.5e-7)
    plt.colorbar()
    plt.show()
    pass


def demo_shtools():
    """
    compare with SHtools
    :return:
    """
    import pyshtools as pysh
    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple

    space_grid = 1
    lat, lon = MathTool.get_global_lat_lon_range(space_grid)
    lmax = 60

    lon, lat = np.meshgrid(lon, lat)

    c, s = load_SH_simple(
        FileTool.get_project_dir(
            'data/L2_SH_products/GSM/CSR/RL06/BA01/2002/GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600'),
        key='GRCOF2',
        lmax=lmax
    )
    c[0, 0] = 0

    topo_sh = np.array([c, s])
    topo_sh = pysh.SHCoeffs.from_array(topo_sh)
    topo_grid = topo_sh.expand(lat=lat, lon=lon)

    shgrid = pysh.SHGrid.from_array(topo_grid)
    topo_sh_new = shgrid.expand(lmax_calc=lmax)

    diff_cs = np.abs(topo_sh.coeffs - topo_sh_new.coeffs)

    diff_cs1d = MathTool.cs_combine_to_triangle_1d(*diff_cs)
    print(np.sum(diff_cs1d ** 2))

    diff_cs_tri = MathTool.cs_combine_to_triangle(*diff_cs)

    for i in range(len(diff_cs_tri)):
        for j in range(len(diff_cs_tri[i])):
            if i < np.abs(j - lmax):
                diff_cs_tri[i, j] = np.nan

    plt.matshow(diff_cs_tri, vmin=0, vmax=1.5e-7)
    plt.colorbar()
    plt.show()

    pass


if __name__ == '__main__':
    demo()
    # demo_shtools()
