import h5py
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


def demo_get_land_mask(resolution=1.):
    from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
    lmax = int(min(360., 180. / resolution))
    grid_space = resolution

    c, s = load_SH_simple(
        FileTool.get_project_dir(
            'data/auxiliary/ocean360_grndline.sh'),
        key='',
        lmax=lmax,
        lmcs_in_queue=(1, 2, 3, 4)
    )

    shc = SHC(c, s)

    grid = shc.to_grid(grid_space)
    grid_data = grid.data[0]
    grid_data[np.where(grid_data > 0.5)] = 1
    grid_data[np.where(grid_data < 0.5)] = 0

    plt.matshow(grid_data)
    plt.colorbar()
    plt.show()

    return grid.lat, grid.lon, 1 - grid_data


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
    # lat_1, lon_1, res_1 = demo_get_land_mask(1)
    # lat_05, lon_05, res_05 = demo_get_land_mask(0.5)

    with h5py.File(FileTool.get_project_dir('temp/20240308/land_mask/GlobalLandMask.hdf5'), 'r') as f:
        pass
    """
    GlobalLandMask.hdf5
      |
      |--resolution_1
      |    |
      |    |--lat
      |    |--lon
      |    |--mask
      |
      |--resolution_05
      |    |
      |    |--lat
      |    |--lon
      |    |--mask
    """

    pass
    # demo_shtools()
