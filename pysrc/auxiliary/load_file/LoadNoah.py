import datetime

import numpy as np
import h5py
from netCDF4 import Dataset

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.data_class.GRD import GRD


class LoadNOAH21:
    def __init__(self):
        self.nc = None
        self.keys = None

    def setFile(self, file):
        self.nc = Dataset(file)
        self.keys = self.nc.variables.keys()
        return self

    def get2dData(self, key, full_lat=False):
        assert key in self.keys, 'no such key'
        data = np.array(self.nc.variables[key])[0]
        xyz = []
        for i in range(len(data)):
            for j in range(len((data[i]))):
                lat = int(i - 90) + 30
                lon = j - 180
                xyz.append([lon, lat, data[i][j]])
        xyz = np.array(xyz)
        grid, lat, lon = MathTool.xyz2grd(xyz)

        if full_lat:
            lat = np.arange(-90 + 0.5, 90 + 0.5, 1)

        return grid, lat, lon


def load_GLDAS_TWS_one_month(filepath, full_lat=True):
    """
    The GLDAS/Noah soil moisture (SM), snow water equivalent (SWE), and plant canopy water storage (PCSW) are jointly
    used to calculate the TWS variations. doi: 10.1155/2019/3874742
    :param filepath: path + filename.nc of NOAH
    :param full_lat:
    :return: 1*1 degree TWS map [m]
    """
    nc = LoadNOAH21().setFile(filepath)
    sm0_10, lat, lon = nc.get2dData('SoilMoi0_10cm_inst', full_lat=full_lat)
    sm10_40, _, _ = nc.get2dData('SoilMoi10_40cm_inst')
    sm40_100, _, _ = nc.get2dData('SoilMoi40_100cm_inst')
    sm100_200, _, _ = nc.get2dData('SoilMoi100_200cm_inst')
    cano, _, _ = nc.get2dData('CanopInt_inst')
    swe, _, _ = nc.get2dData('SWE_inst')
    return (sm0_10 + sm10_40 + sm40_100 + sm100_200 + cano + swe) / 1000, lat, lon


def load_GLDAS_TWS(begin_date: datetime.date = None, end_date: datetime.date = None, from_exist_results=None,
                   de_average=True, log=False, full_lat=True):
    """
    :param begin_date: datetime.date
    :param end_date: datetime.date
    :param from_exist_results: hdf5 file
    :param de_average: bool,
    :param log: bool,
    :param full_lat: bool,
    :return: GRID, times
    """
    assert not from_exist_results

    filedir = FileTool.get_project_dir(relative=True) / 'data/Noah2.1'
    filepaths_list = list(filedir.iterdir())
    filepaths_list.sort()

    grids = []
    times = []
    lat, lon = None, None
    for i in range(len(filepaths_list)):
        filename = filepaths_list[i].name
        yyyymm = filename.split('_')[2][3:9]
        year = int(yyyymm[:4])
        month = int(yyyymm[4:])

        this_date = datetime.date(year, month, 15)

        if begin_date <= this_date <= end_date:
            if log:
                print(f'calculating: {filename}...')
            this_tws, this_lat, this_lon = load_GLDAS_TWS_one_month(filedir / filename, full_lat=full_lat)

            if lat is None:
                lat = this_lat
            if lon is None:
                lon = this_lon

            grids.append(this_tws)
            times.append(this_date)

    if de_average:
        grids -= np.mean(grids, axis=0)

    return GRD(grids, lat=lat, lon=lon), times
