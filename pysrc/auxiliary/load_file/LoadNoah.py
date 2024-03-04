import datetime

import numpy as np
import h5py
from netCDF4 import Dataset

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool


class LoadNOAH21:
    def __init__(self):
        self.nc = None
        self.keys = None

    def setFile(self, file):
        self.nc = Dataset(file)
        self.keys = self.nc.variables.keys()
        return self

    def get2dData(self, key):
        assert key in self.keys, 'no such key'
        data = np.array(self.nc.variables[key])[0]
        xyz = []
        for i in range(len(data)):
            for j in range(len((data[i]))):
                lat = int(i - 90) + 30
                lon = j - 180
                xyz.append([lon, lat, data[i][j]])
        xyz = np.array(xyz)
        grid = MathTool.xyz2grd(xyz)
        return grid


def getTWS_one_month(filepath):
    """
    The GLDAS/Noah soil moisture (SM), snow water equivalent (SWE), and plant canopy water storage (PCSW) are jointly
    used to calculate the TWS variations. doi: 10.1155/2019/3874742
    :param filepath: path + filename.nc of NOAH
    :return: 1*1 degree TWS map [m]
    """
    nc = LoadNOAH21().setFile(filepath)
    sm0_10 = nc.get2dData('SoilMoi0_10cm_inst')
    sm10_40 = nc.get2dData('SoilMoi10_40cm_inst')
    sm40_100 = nc.get2dData('SoilMoi40_100cm_inst')
    sm100_200 = nc.get2dData('SoilMoi100_200cm_inst')
    cano = nc.get2dData('CanopInt_inst')
    swe = nc.get2dData('SWE_inst')
    return (sm0_10 + sm10_40 + sm40_100 + sm100_200 + cano + swe) / 1000


def get_TWS_series(begin_date: datetime.date = None, end_date: datetime.date = None, from_exist_results=None,
                   de_average=True):
    """
    :param begin_date: datetime.date
    :param end_date: datetime.date
    :param from_exist_results: hdf5 file
    :param de_average: bool
    """

    if from_exist_results is None:
        filedir = FileTool.get_project_dir(relative=True) / 'data/Noah2.1'
        filepaths_list = list(filedir.iterdir())

        grids = []
        times = []
        for i in range(len(filepaths_list)):
            filename = filepaths_list[i].name
            yyyymm = filename.split('_')[2][3:9]
            year = int(yyyymm[:4])
            month = int(yyyymm[4:])

            this_date = datetime.date(year, month, 1)

            if begin_date <= this_date <= end_date:
                print(f'calculating: {filename}...')
                this_tws = getTWS_one_month(filedir / filename)

                grids.append(this_tws)
                times.append(datetime.date(year, month, 1))

    else:
        with h5py.File(from_exist_results, 'r') as f:
            times_mjd = np.array(f['time'])
            times = TimeTool.convert_date_format(times_mjd,
                                                 input_type=TimeTool.DateFormat.MJD,
                                                 output_type=TimeTool.DateFormat.ClassDate)

            grids = np.array(f['EWH'])

    if de_average:
        grids -= np.mean(grids, axis=0)

    return grids, times


def demo():
    from pathlib import Path
    from pysrc.post_processing.harmonic.Harmonic import Harmonic

    import cartopy.crs as ccrs
    import cartopy.feature as cft

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import cmaps

    filepath = FileTool.get_project_dir(relative=True) / 'data/Noah2.1'
    filepaths_list = list(filepath.iterdir())

    grids = []
    times = []
    for i in range(len(filepaths_list)):
        filename = filepaths_list[i].name
        yyyymm = filename.split('_')[2][3:9]
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])

        if 2008 <= year <= 2008:
            print(f'calculating: {filename}...')
            this_tws = getTWS_one_month(filepath / filename)

            grids.append(this_tws)
            times.append(datetime.date(year, month, 1))

    print('harmonic analysing...')
    grids = np.array(grids)
    minimum = np.min(grids)
    grids[np.where(grids == 0)] = minimum

    lmax = 60
    grid_space = 1

    lat = np.arange(-90 + grid_space / 2, 90 + grid_space / 2, grid_space)
    lon = np.arange(-180 + grid_space / 2, 180 + grid_space / 2, grid_space)

    fig = plt.figure(figsize=(10, 10))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    lon2d, lat2d = np.meshgrid(lon, lat)
    ax.pcolormesh(lon2d, lat2d, grids[0], norm=norm)
    ax.add_feature(cft.COASTLINE)
    plt.show()

    har = Harmonic(lat, lon, lmax, option=1)


if __name__ == '__main__':
    # begin = datetime.date(2002, 4, 1)
    # end = datetime.date(2023, 7, 31)
    #
    # grids, times = get_TWS_series(begin, end)
    #
    # dict_to_save = dict(
    #     EWH=np.array(grids),
    #     time=np.array(TimeTool.convert_date_format(times,
    #                                                input_type=TimeTool.DateFormat.ClassDate,
    #                                                output_type=TimeTool.DateFormat.MJD)
    #                   ))
    #
    # make_hdf5(FileTool.get_project_dir() / 'results/NOAH_EWH/NOAH_EWH_200204_202307.hdf5', dict_to_save)

    grids, times = get_TWS_series(
        from_exist_results=FileTool.get_project_dir() / 'results/NOAH_EWH/NOAH_EWH_200204_202307.hdf5')
