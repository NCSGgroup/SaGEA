#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/12/29 15:06 
# @File    : core.py
import copy
import datetime
import pathlib

import numpy
import numpy as np
from pathlib import Path
import warnings

from sagea import constant
from sagea.processing.geometric_correction.GeometricalCorrection import GeometricalCorrection
from sagea.sgio.gfc_reader import read_gfc
from sagea.processing.Harmonic import Harmonic, GRDType
from sagea.processing.filter.GetSHCFilter import get_filter
from sagea.utils import MathTool, TimeTool

from sagea.processing.SHCPhysicalConvert import ConvertSHC
from sagea.processing.LoveNumber import LoveNumber


class SHC:
    """Spherical Harmonic coefficients"""

    # --- generate SHC instance --- #
    # Following methods will generate and return SHC an instance class
    def __init__(self, cs: numpy.ndarray, normalized="4pi"):
        assert cs.ndim in (1, 2)
        assert normalized in ("4pi",)

        self.value = cs if cs.ndim == 2 else cs[None, :]
        lmax_approx = np.sqrt(self.value.shape[1]) - 1
        assert np.isclose(lmax_approx, np.round(lmax_approx), atol=1e-8), "Invalid shape for CS arrays"

        self.normalized = normalized

    @staticmethod
    def from_file(filepath: Path or list[Path], lmax, key="gfc", cols=None, normalized="4pi"):
        """
        load SHC from filepath or list[filepath]
        """
        assert isinstance(filepath, Path) or isinstance(filepath, list)

        if isinstance(filepath, Path):
            cs_array = read_gfc(filepath, key=key, lmax=lmax, col_indices=cols)
        elif isinstance(filepath, list):
            cs_array = []
            for path in filepath:
                cs = read_gfc(path, key=key, lmax=lmax, col_indices=cols)
                cs_array.append(cs)
            cs_array = numpy.array(cs_array)
        else:
            assert False

        return SHC(cs_array, normalized=normalized)

    @staticmethod
    def from_trend(shc_trend, times: list[datetime.date], ref_time: datetime.date = None):
        """
        Generate a list of SHC instances by linearly propagating a trend SHC.

        The coefficients are calculated as:
            C(t) = C_rate * (t - t_ref)

        Parameters
        ----------
        shc_trend : SHC
            An SHC instance containing the trend rates (e.g., GIA trend in /year).
        times : list of datetime.date
            The list of dates for which to generate the SHCs.
        ref_time : datetime.date, optional
            The reference epoch where the signal is zero.
            If None, defaults to the first date in `times`.

        Returns
        -------
        list of SHC
            A list of new SHC objects, one for each target date.
        """
        assert len(shc_trend) == 1

        if ref_time is None:
            ref_time = times[0]

        year_frac = TimeTool.convert_date_format(times,
                                                 input_type=TimeTool.DateFormat.ClassDate,
                                                 output_type=TimeTool.DateFormat.YearFraction)
        year_frac = np.array(year_frac)
        year_frac -= TimeTool.convert_date_format(ref_time,
                                                  input_type=TimeTool.DateFormat.ClassDate,
                                                  output_type=TimeTool.DateFormat.YearFraction)

        year_frac = np.array(year_frac)

        trend = shc_trend.value[0]
        value = year_frac[:, None] @ trend[None, :]
        return SHC(value)

    # --- properties and statistics --- #
    # Following methods return the required information about this instance.
    @property
    def lmax(self):
        return int(np.sqrt(self.value.shape[1]) - 1)

    def __len__(self):
        return self.value.shape[0]

    @property
    def cs2d(self):
        """
        tuple:
            cqlm, sqlm.
            Both cqlm and sqlm are 3-dimension, EVEN IF self.__len__() > 1
        """
        lmax = self.lmax
        num_of_series = np.shape(self.value)[0]

        cqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))
        sqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))

        for i in range(num_of_series):
            this_cs = self.value[i]
            this_clm, this_slm = MathTool.cs_decompose_triangle1d_to_cs2d(this_cs)
            cqlm[i, :, :] = this_clm
            sqlm[i, :, :] = this_slm

        return cqlm, sqlm

    @property
    def mean(self):
        return np.mean(self.value, axis=0)

    @property
    def get_std(self):
        return np.std(self.value, axis=0)

    @property
    def get_var(self):
        return np.cov(self.value.T)

    @property
    def degree_rms(self):
        cqlm, sqlm = self.cs2d
        return MathTool.get_degree_rms(cqlm, sqlm)

    @property
    def degree_rss(self):
        cqlm, sqlm = self.cs2d
        return MathTool.get_degree_rss(cqlm, sqlm)

    @property
    def cumulative_degree_rss(self):
        cqlm, sqlm = self.cs2d
        return MathTool.get_cumulative_rss(cqlm, sqlm)

    # --- calculation and processing --- #
    # Following methods process and change the values in the instance.
    def __add__(self, other):
        assert isinstance(other, SHC)
        assert self.lmax == other.lmax

        return SHC(self.value + other.value)

    def __sub__(self, other):
        assert isinstance(other, SHC)
        assert self.lmax == other.lmax

        return SHC(self.value - other.value)

    def replace(self, index: str, new: np.ndarray):
        """

        :param index: str, like "c2,0" or "s1,1"
        :param new:
        :return:
        """
        assert isinstance(new, SHC) or isinstance(new, np.ndarray)

        index1d = MathTool.get_cs_1d_index(index)

        if isinstance(new, SHC):
            new_array = new.value[:, index1d]
        else:
            new_array = new

        assert len(new_array) == len(self)

        index_new_array_valid = np.where(new_array == new_array)  # ignore np.nan

        self.value[index_new_array_valid, index1d] = new_array[index_new_array_valid]

    def de_mean(self):
        self.value -= self.mean

        return self

    def filter(self, method: constant.SHCFilterType or constant.SHCDecorrelationType, param: tuple = None):
        cqlm, sqlm = self.cs2d
        filtering = get_filter(method, param, lmax=self.lmax)
        cqlm_f, sqlm_f = filtering.apply_to(cqlm, sqlm)
        self.value = MathTool.cs_combine_to_triangle_1d(cqlm_f, sqlm_f)

        return self

    def convert_physical(self, from_type=None, to_type=None):
        if from_type is None:
            from_type = constant.PhysicalDimension.Geopotential
        if to_type is None:
            to_type = constant.PhysicalDimension.Geopotential

        assert from_type in constant.PhysicalDimension
        assert to_type in constant.PhysicalDimension

        lmax = self.lmax
        LN = LoveNumber()
        LN.configuration.set_lmax(lmax)
        ln = LN.get_Love_number()

        convert = ConvertSHC()
        convert.configuration.set_Love_number(ln).set_input_type(from_type).set_output_type(to_type)

        self.value = convert.apply_to(self.value)
        return self

    def geometric(self, assumption: constant.GeometricCorrectionAssumption, log=False):
        gc = GeometricalCorrection()
        cqlm, sqlm = self.cs2d
        cqlm_new, sqlm_new = gc.apply_to(cqlm, sqlm, assumption=assumption, log=log)
        self.value = MathTool.cs_combine_to_triangle_1d(cqlm_new, sqlm_new)

        return self

    # --- harmonic synthesis --- #
    # Following methods generate a GRD instance or numpy arrays of spatial data.
    def to_GRD(self, grid_space=None, grid_type: GRDType or None = None):
        """pure synthesis"""
        if grid_type is None and grid_space is None:
            grid_type = GRDType.GLQ

        if (grid_type is not None) and (grid_space is not None):
            warnings.warn(
                "Both 'grid_space' and 'grid_type' were provided. "
                "'grid_space' takes precedence, and 'grid_type' will be set to None.",
                category=UserWarning,
                stacklevel=2
            )
            grid_type = None

        lmax = self.lmax
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)
        har = Harmonic(lmax=lmax, grid_type=grid_type, lat=lat, lon=lon)

        cqlm, sqlm = self.cs2d
        grid_data = har.synthesis(cqlm, sqlm)

        grid = GRD(grid_data, har.colat, har.lon, option=0)
        grid.grid_type = grid_type

        return grid


class GRD:
    """Gridded value"""

    # --- generate GRD instance --- #
    # Following methods will generate and return GRD an instance class
    def __init__(self, grid, lat, lon, option=1):
        """
        To create a GRID object,
        one needs to specify the data (grid) and corresponding latitude range (lat) and longitude range (lon).
        :param grid: 2d- or 3d-array gridded signal, index ([num] ,lat, lon)
        :param lat: co-latitude in [rad] if option=0 else latitude in [degree]
        :param lon: longitude in [rad] if option=0 else longitude in [degree]
        :param option: set 0 if input colat and lon are in [rad]
        """
        if np.ndim(grid) == 2:
            grid = [grid]
        assert np.shape(grid)[-2:] == (len(lat), len(lon))

        self.value = np.array(grid)

        self.__grid_type = None

        if option == 0:
            self.lat = 90 - np.degrees(lat)
            self.lon = np.degrees(lon)

        else:
            self.lat = lat
            self.lon = lon

        self.dates_series = None

    # --- properties and statistics --- #
    # Following methods return the required information about this instance.
    @property
    def grid_type(self):
        return self.__grid_type

    def regional_extraction(self, mask: np.ndarray, average=True, leakage: constant.LeakageMethod = None, **kwargs):
        """
        Extract signal over a specific region with optional leakage correction.

        Parameters
        ----------
        mask : np.ndarray
            Region mask (0/1 or boolean), matching grid dimensions.
        average : bool, default=True
            get average signal over the region (i.e., divided by regional size) if True else Total signal.
        leakage : str, optional
            Method for leakage correction. Options:
            - None (Default): Direct integration.
            - 'additive':
            - 'multiplicative':
            - 'scaling':
            - 'scaling_grid':
            - 'data_driven':
            - 'forward_modeling': Iterative FM. Accepts 'max_iter'.
            - 'buffer_shrink':
            - 'buffer_expand':
        **kwargs :
            Parameters specific to the chosen correction method.

        Returns
        -------
        numpy.array[float]
            Total/average value in the region (e.g., Gt or EWH sum).
        """

        from sagea.processing.leakage.Additive import additive
        from sagea.processing.leakage.Multiplicative import multiplicative
        from sagea.processing.leakage.Scaling import scaling
        from sagea.processing.leakage.ScalingGrid import scaling_grid
        from sagea.processing.leakage.ForwardModeling import forward_modeling
        from sagea.processing.leakage.DataDriven import data_driven
        from sagea.processing.leakage.BufferZone import buffer_zone

        assert isinstance(mask, np.ndarray)

        dispatch = {
            None: None,
            constant.LeakageMethod.Additive: additive,
            constant.LeakageMethod.Multiplicative: multiplicative,
            constant.LeakageMethod.Scaling: scaling,
            constant.LeakageMethod.ScalingGrid: scaling_grid,
            constant.LeakageMethod.DataDriven: data_driven,
            constant.LeakageMethod.ForwardModeling: forward_modeling,
            constant.LeakageMethod.BufferZone: buffer_zone,
        }

        leakage_corrector = dispatch[leakage]

        if leakage_corrector is not None:
            assert average, "can only set average=True for leakage not None"

        lat, lon = self.lat, self.lon

        if leakage_corrector is not None:
            integral_result = leakage_corrector(grid_value=self.value, lat=lat, lon=lon, basin_mask=mask, **kwargs)

        else:
            integral_result = MathTool.global_integral(self.value * mask, lat, lon)
            if average:
                integral_result /= MathTool.get_acreage(mask)

        return integral_result

    # --- calculation and processing --- #
    # Following methods process and change the values in the instance.
    @grid_type.setter
    def grid_type(self, grid_type: GRDType):
        self.__grid_type = grid_type

    def limiter(self, threshold=0.5, beyond=1, below=0):
        index_beyond = np.where(self.value >= threshold)
        index_below = np.where(self.value < threshold)

        self.value[index_beyond] = beyond
        self.value[index_below] = below
        return self

    def de_aliasing(self, times, s2=False, p1=False, s1=False, k2=False, k1=False):
        """

        """

        from sagea.processing.DeAliasing import DeAliasing
        de_alias = DeAliasing()

        de_alias.configuration.set_de_s2(s2),
        de_alias.configuration.set_de_p1(p1),
        de_alias.configuration.set_de_s1(s1),
        de_alias.configuration.set_de_k2(k2),
        de_alias.configuration.set_de_k1(k1),

        year_frac = TimeTool.convert_date_format(
            times, input_type=TimeTool.DateFormat.ClassDate, output_type=TimeTool.DateFormat.YearFraction
        )

        self.value = de_alias.apply_to(self.value, year_frac)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import cartopy.crs
    import matplotlib
    from sagea.sgio.low_deg_reader import read_low_degs

    file_path_list = list(
        pathlib.Path(
            "/Users/shuhao/PycharmProjects/SaGEA_update/data/L2_SH_products/GSM/CSR/RL06/BA01/2005"
        ).iterdir())
    file_path_list.sort()

    dates_begin, dates_end = TimeTool.match_dates_from_name(file_path_list)
    dates_ave = TimeTool.get_average_dates(dates_begin, dates_end)

    path_tn14 = pathlib.Path(
        "/Users/shuhao/PycharmProjects/SaGEA_update/data/L2_low_degrees/TN-14_C30_C20_SLR_GSFC.txt"
    )
    path_tn13 = pathlib.Path(
        "/Users/shuhao/PycharmProjects/SaGEA_update/data/L2_low_degrees/TN-13_GEOC_JPL_RL06.txt"
    )

    low_degs = read_low_degs(path_tn14, dates_ave)
    low_degs.update(read_low_degs(path_tn13, dates_ave))

    shc = SHC.from_file(file_path_list, lmax=60, key="GRCOF2")
    keys_low_deg = ["c1,0", "c1,1", "s1,1", "c2,0", "c3,0"]
    for key in keys_low_deg:
        shc.replace(key, low_degs[key])

    path_gia = pathlib.Path(
        "/Users/shuhao/PycharmProjects/SaGEA_update/data/GIA/GIA.ICE-6G_D.txt"
    )
    shc_gia_trend = SHC.from_file(path_gia, lmax=60, key="")
    shc_gia = SHC.from_trend(shc_gia_trend, dates_ave)

    shc -= shc_gia

    shc.de_mean()

    shc_unf = copy.deepcopy(shc)
    shc.filter(constant.SHCFilterType.HAN, (200, 300, 30,))

    # shc.geometric(assumption=constant.GeometricCorrectionAssumption.Ellipsoid, log=False)

    shc.convert_physical(
        from_type=constant.PhysicalDimension.Geopotential,
        to_type=constant.PhysicalDimension.EWH
    )
    shc_unf.convert_physical(
        from_type=constant.PhysicalDimension.Geopotential,
        to_type=constant.PhysicalDimension.EWH
    )

    grd = shc.to_GRD(1)

    ref = copy.deepcopy(grd.value)

    shc_basin_path = pathlib.Path(
        "/Users/shuhao/PycharmProjects/SaGEA_update/data/basin_mask/SH/Brahmaputra_maskSH.dat"
    )
    shc_basin = SHC.from_file(shc_basin_path, lmax=60, key="")
    grd_basin = shc_basin.to_GRD(1)
    grd_basin.limiter()
    mask = grd_basin.value[0]

    shc_basin_ocean_path = pathlib.Path(
        "/Users/shuhao/PycharmProjects/SaGEA_update/data/basin_mask/SH/Ocean_maskSH.dat"
    )
    shc_basin_ocean = SHC.from_file(shc_basin_ocean_path, lmax=60, key="")
    grd_basin_ocean = shc_basin_ocean.to_GRD(1)
    grd_basin_ocean.limiter()
    mask_conservation_for_fm = grd_basin_ocean.value[0]

    y = grd.regional_extraction(mask=mask, average=True, leakage=None)

    '''leakage correction'''
    # y_corrected = grd.regional_extraction(mask=mask, average=True,
    #                                       leakage=constant.LeakageMethod.ScalingGrid,
    #                                       reference=ref[:-1],
    #                                       filter_method=constant.SHCFilterType.HAN,
    #                                       filter_param=(200, 300, 30,),
    #                                       lmax_calc=60
    #                                       )

    # y_corrected = grd.regional_extraction(mask=mask, average=True,
    #                                       leakage=constant.LeakageMethod.ForwardModeling,
    #                                       basin_conservation=mask_conservation_for_fm,
    #                                       filter_method=constant.SHCFilterType.HAN,
    #                                       filter_param=(200, 300, 30,),
    #                                       lmax_calc=60
    #                                       )

    # y_corrected = grd.regional_extraction(mask=mask, average=True,
    #                                       leakage=constant.LeakageMethod.DataDriven,
    #                                       shc_unfiltered=shc_unf,
    #                                       filter_method=constant.SHCFilterType.HAN,
    #                                       filter_param=(200, 300, 30,),
    #                                       lmax_calc=60,
    #                                       )

    # y_corrected = grd.regional_extraction(mask=grd_basin_ocean.value[0], average=True,
    #                                       leakage=constant.LeakageMethod.BufferZone,
    #                                       filter_method=constant.SHCFilterType.HAN,
    #                                       filter_param=(200, 300, 30,),
    #                                       lmax_calc=60,
    #                                       buffer_type="shrink",
    #                                       threshold=0.1
    #                                       )

    '''de aliasing'''
    grd.de_aliasing(times=dates_ave, s2=True)
    y_corrected = grd.regional_extraction(mask=mask, average=True, leakage=None)

    x = TimeTool.convert_date_format(dates_ave)

    plt.plot(x, y, label="uncorrected")
    plt.plot(x, y_corrected, label="corrected")
    plt.legend()
    plt.show()

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=cartopy.crs.Robinson())
    #
    # lon2d, lat2d = np.meshgrid(grd.lon, grd.lat)
    # ax.pcolormesh(
    #     lon2d, lat2d, grd.value[10] * 100,
    #     transform=cartopy.crs.PlateCarree(),
    #     norm=matplotlib.colors.TwoSlopeNorm(vmin=-20, vmax=20, vcenter=0),
    #     # zorder=2
    # )
    #
    # ax.add_feature(cartopy.feature.COASTLINE)
    #
    # plt.show()
