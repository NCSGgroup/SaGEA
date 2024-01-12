import copy
import enum
import json
import pathlib
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np

from pysrc.auxiliary.core.GRID import GRID
from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.load_file.LoadGIA import LoadGIA
from pysrc.auxiliary.load_file.LoadL2LowDeg import LoadLowDegree
from pysrc.auxiliary.load_file.LoadNoah import get_TWS_series
from pysrc.auxiliary.load_file.LoadShp import ShpToMask
from pysrc.auxiliary.preference.EnumClasses import L2LowDegreeFileID, L2InstituteType, L2Release, SHCDecorrelationType, \
    SHCFilterType, LeakageMethod, GIAModel, BasinName, SHCDecorrelationSlidingWindowType, L2ProductType
from pysrc.auxiliary.scripts.MatchConfigWithEnums import match_config
from pysrc.auxiliary.tools.FileTool import FileTool
from pysrc.auxiliary.tools.MathTool import MathTool
from pysrc.auxiliary.tools.TimeTool import TimeTool
from pysrc.post_processing.GIA_correction.GIACorrectionSpectral import GIACorrectionSpectral
from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC, FieldPhysicalQuantity
from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH, load_SH_simple
from pysrc.auxiliary.scripts.PlotGrids import plot_grids
from pysrc.post_processing.filter.DDK import DDK, DDKFilterType
from pysrc.post_processing.filter.GetSHCFilter import get_shc_decorrelation, get_shc_filter
from pysrc.post_processing.leakage.Addictive import Addictive
from pysrc.post_processing.leakage.BaseModelDriven import ModelDriven
from pysrc.post_processing.leakage.BufferZone import BufferZone
from pysrc.post_processing.leakage.GetLeakageDeductor import get_leakage_deductor
from pysrc.post_processing.leakage.Multiplicative import Multiplicative

from pysrc.post_processing.harmonic.Harmonic import Harmonic

import datetime

from pysrc.post_processing.leakage.Scaling import Scaling
from pysrc.post_processing.leakage.ScalingGrid import ScalingGrid
from pysrc.post_processing.replace_low_deg.ReplaceLowDegree import ReplaceLowDegree


class PostProcessingConfig:
    def __init__(self):
        self.__begin_date = datetime.date(2005, 1, 1)
        self.__end_date = datetime.date(2015, 12, 31)
        self.__basin = BasinName.Amazon

        self.__GRACE_institute = L2InstituteType.CSR
        self.__GRACE_release = L2Release.RL06

        self.__lmax = 60

        self.__grid_space = 1

        self.__replace_low_degree_coefficients = ('degree1', 'c20', 'c30')
        self.__low_degree_degree1_file_id = L2LowDegreeFileID.TN13
        self.__low_degree_c20_file_id = L2LowDegreeFileID.TN14
        self.__low_degree_c30_file_id = L2LowDegreeFileID.TN14

        self.__de_correlation_method = SHCDecorrelationType.PnMm
        self.__de_correlation_sliding_window_type = SHCDecorrelationSlidingWindowType.Wahr2006
        self.__de_correlation_params = (3, 5,)

        self.__shc_filter = SHCFilterType.Gaussian
        self.__shc_filter_params = (300,)

        self.__leakage_type = LeakageMethod.Scaling

        self.__GIA_model = GIAModel.ICE6GD

    def set_from_json(self, filepath: str or pathlib.WindowsPath or dict):
        """
        :param filepath: json filepath, or a dict loaded from json.
        """
        assert type(filepath) in (str, pathlib.WindowsPath, dict)

        if type(filepath) in (str, pathlib.WindowsPath):
            with open(filepath, 'r') as f:
                dict_from_jason = json.load(f)

        elif type(filepath) in (dict,):
            dict_from_jason = filepath

        else:
            return -1

        assert ({'begin_date',
                 'end_date',
                 'basin',
                 'GRACE_institute',
                 'GRACE_release',
                 'lmax',
                 'replace_low_degree',
                 'low_degree_degree1_file_id',
                 'low_degree_c20_file_id',
                 'low_degree_c30_file_id',
                 'de_correlation',
                 'de_correlation_sliding_window_type',
                 'de_correlation_params',
                 'shc_filter',
                 'shc_filter_params',
                 'leakage_method',
                 'GIA_model'} <= set(dict_from_jason.keys()))

        self.set_begin_date(dict_from_jason['begin_date'])
        self.set_end_date(dict_from_jason['end_date'])
        self.set_lmax(dict_from_jason['lmax'])
        self.set_replace_low_degree_coefficients(dict_from_jason['replace_low_degree'])
        self.set_de_correlation_params(dict_from_jason['de_correlation_params'])

        self.set_shc_filter_params(dict_from_jason['shc_filter_params'])

        enum_classes = [
            L2InstituteType, L2Release, L2LowDegreeFileID, L2LowDegreeFileID, L2LowDegreeFileID,
            SHCDecorrelationType, SHCDecorrelationSlidingWindowType, SHCFilterType, LeakageMethod, GIAModel
        ]

        json_keys = [
            'GRACE_institute', 'GRACE_release', 'low_degree_degree1_file_id', 'low_degree_c20_file_id',
            'low_degree_c30_file_id', 'de_correlation', 'de_correlation_sliding_window_type', 'shc_filter',
            'leakage_method', 'GIA_model'
        ]

        setting_functions = [
            self.set_GRACE_institute, self.set_GRACE_release, self.set_degree1_file_id,
            self.set_c20_file_id, self.set_c30_file_id, self.set_de_correlation_method,
            self.set_de_correlation_sliding_window_mode, self.set_shc_filter_method, self.set_leakage_method,
            self.set_GIA_model
        ]

        if dict_from_jason['basin'] in BasinName.__members__.keys():
            enum_classes.append(BasinName)
            json_keys.append('basin')
            setting_functions.append(self.set_basin)

        else:
            self.set_basin(pathlib.Path(dict_from_jason['basin']))

        match_config(dict_from_jason, json_keys, enum_classes, setting_functions)

        return self

    def export_json(self, filepath: str or pathlib.WindowsPath):
        pass

    def set_begin_date(self, d):

        if type(d) is datetime.date:
            self.__begin_date = d

        elif type(d) is str:
            re_match = re.match(r'(\d{4})-(\d{1,2})-*(\d{1,2})*', d)
            year, month, day = re_match.groups()
            if day is None:
                day = 1

            self.__begin_date = datetime.date(int(year), int(month), int(day))

        else:
            return -1

        return self

    def get_begin_date(self):
        return self.__begin_date

    def set_end_date(self, d):

        if type(d) is datetime.date:
            self.__end_date = d

        elif type(d) is str:
            re_match = re.match(r'(\d{4})-(\d{1,2})-*(\d{1,2})*', d)
            year, month, day = re_match.groups()

            if day is None:
                day = TimeTool.get_the_final_day_of_this_month(datetime.date(int(year), int(month), 1)).day

            self.__end_date = datetime.date(int(year), int(month), int(day))

        else:
            return -1

        return self

    def get_end_date(self):
        return self.__end_date

    def set_basin(self, basin: BasinName or pathlib.WindowsPath or str):
        """
        :param basin: a preset BasinName class or a pathlib.WindowsPath or str that describes a relative filepath.
                    if a path, it should be a spherical harmonic coefficient file to describe the kernel, or a shpfile,
                    and the filepath should be related to '*project_dir*/'
        """
        self.__basin = basin

        return self

    def get_basin(self):
        return self.__basin

    def set_GRACE_institute(self, institute: L2InstituteType or str):
        assert type(institute) in (L2InstituteType, str)
        if type(institute) is L2InstituteType:
            self.__GRACE_institute = institute
        else:
            self.__GRACE_institute = L2InstituteType[institute]

        return self

    def get_GRACE_institute(self):
        return self.__GRACE_institute

    def set_GRACE_release(self, release: L2Release):
        self.__GRACE_release = release

        return self

    def get_GRACE_release(self):
        return self.__GRACE_release

    def set_lmax(self, lmax: int):
        self.__lmax = lmax

        return self

    def get_lmax(self):
        return int(self.__lmax)

    def set_grid_space(self, gs: int):
        self.__grid_space = gs

        return self

    def get_grid_space(self):
        return self.__grid_space

    def set_replace_low_degree_coefficients(self, replace_coefficients):
        """
        :param replace_coefficients: a tuple that contains (or not) "degree1", "c20", "c30".
        """
        self.__replace_low_degree_coefficients = replace_coefficients

        return self

    def get_replace_low_degree_coefficients(self):
        return self.__replace_low_degree_coefficients

    def set_degree1_file_id(self, file_id: L2LowDegreeFileID):
        self.__low_degree_degree1_file_id = file_id

        return self

    def get_degree1_file_id(self):
        return self.__low_degree_degree1_file_id

    def set_c20_file_id(self, file_id: L2LowDegreeFileID):
        self.__low_degree_c20_file_id = file_id

        return self

    def get_c20_file_id(self):
        return self.__low_degree_c20_file_id

    def set_c30_file_id(self, file_id: L2LowDegreeFileID):
        self.__low_degree_c30_file_id = file_id

        return self

    def get_c30_file_id(self):
        return self.__low_degree_c30_file_id

    def set_de_correlation_method(self, method: SHCDecorrelationType):
        self.__de_correlation_method = method

        return self

    def get_de_correlation_method(self):
        return self.__de_correlation_method

    def set_de_correlation_params(self, params: tuple):
        """
        :param params: (n, m) for PnMm method;
                    (n, m, window length) for sliding window (stable) method;
                    (n, m, minimize window length, A, K) for sliding window (Wahr2006) method;
        """
        self.__de_correlation_params = params

        return self

    def get_de_correlation_params(self):
        return self.__de_correlation_params

    def set_de_correlation_sliding_window_mode(self, method: SHCDecorrelationSlidingWindowType):
        self.__de_correlation_sliding_window_type = method

        return self

    def get_de_correlation_sliding_window_mode(self):
        return self.__de_correlation_sliding_window_type

    def set_shc_filter_method(self, method: SHCFilterType):
        self.__shc_filter = method

        return self

    def get_shc_filter_method(self):
        return self.__shc_filter

    def set_shc_filter_params(self, params: tuple):
        """
        :param params: (radius[km], ) for Gaussian,
                    (radius_1[km], radius_2[km]) for Fan,
                    (radius_1[km], radius_2[km], m_0) for AnisotropicGaussianHan,
                    (DDKFilterType, ) for DDK
        """

        self.__shc_filter_params = params

        return self

    def get_shc_filter_params(self):
        return self.__shc_filter_params

    def set_leakage_method(self, method: LeakageMethod):
        self.__leakage_type = method

        return self

    def get_leakage_method(self):
        return self.__leakage_type

    def set_GIA_model(self, model: GIAModel):
        self.__GIA_model = model

        return self

    def get_GIA_model(self):
        return self.__GIA_model


class PostProcessing:
    def __init__(self):
        self.configuration = PostProcessingConfig()

        self.times = None
        self.time_series_ewh = []

        self.shc_GRACE = None
        self.filtered_shc = None
        self.shc_basin = None

        self.grid = None
        self.filtered_grid = None

        self.basin_map = None

        self.harmonic = None
        self.shc_filter = None

        self.auxiliary_data = {}

    def prepare(self):
        lmax = self.configuration.get_lmax()
        grid_space = self.configuration.get_grid_space()

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)
        har = Harmonic(lat, lon, lmax, option=1)

        self.harmonic = har

        return self

    def load_files(self):
        self.__load_basin()
        self.__load_GRACE_shc_and_replace_low_degree()

        return self

    def __load_basin(self):

        lmax = self.configuration.get_lmax()

        basin = self.configuration.get_basin()

        if isinstance(basin, BasinName):
            basin_name = basin.name
            basin_shc_filepath = FileTool.get_project_dir() / f'data/basin_mask/{basin_name}_maskSH.dat'
            basin_clm, basin_slm = load_SH_simple(basin_shc_filepath, key='', lmcs_in_queue=(1, 2, 3, 4), lmax=lmax)
            shc_basin = SHC(basin_clm, basin_slm)

            basin_map = self.harmonic.synthesis(shc_basin).data

        elif type(basin) is pathlib.WindowsPath:
            shp_filepath = FileTool.get_project_dir() / basin

            grid_space = self.configuration.get_grid_space()

            load_shp = ShpToMask()
            load_shp.configuration.set_grid_space(grid_space)

            load_shp.configuration.set_shppath(shp_filepath)

            basin_map = load_shp.get_basin_gridmap(with_whole=True)

            basin_grid = GRID(basin_map, lat=self.harmonic.lat, lon=self.harmonic.lon)
            shc_basin = self.harmonic.analysis(basin_grid)

        else:
            return -1

        self.shc_basin = shc_basin
        self.basin_map = basin_map

        return self

    def __load_GRACE_shc_and_replace_low_degree(self, background="average"):
        """
        :param background: average, None or class SHC
        """

        '''load GRACE L2 SH products'''
        begin_date, end_date = self.configuration.get_begin_date(), self.configuration.get_end_date()
        institute = self.configuration.get_GRACE_institute()
        lmax = self.configuration.get_lmax()

        load = LoadL2SH()

        load.configuration.set_begin_date(begin_date)
        load.configuration.set_end_date(end_date)
        load.configuration.set_institute(institute)
        load.configuration.set_lmax(lmax)

        shc, dates = load.get_shc(with_dates=True)
        ave_dates_GRACE = TimeTool.get_average_dates(*dates)

        '''load and replace low degrees'''
        degree1_or_not = 'degree1' in self.configuration.get_replace_low_degree_coefficients()
        degree1_file_id = self.configuration.get_degree1_file_id()

        c20_or_not = 'c20' in self.configuration.get_replace_low_degree_coefficients()
        c20_file_id = self.configuration.get_c20_file_id()

        c30_or_not = 'c30' in self.configuration.get_replace_low_degree_coefficients()
        c30_file_id = self.configuration.get_c30_file_id()

        low_degs = {}

        if degree1_or_not:
            load_deg1 = LoadLowDegree()
            load_deg1.configuration.set_file_id(degree1_file_id).set_institute(institute)
            low_degs.update(load_deg1.get_degree1())

        if c20_or_not:
            load_c20 = LoadLowDegree()
            load_c20.configuration.set_file_id(c20_file_id)
            low_degs.update(load_c20.get_c20())

        if c30_or_not:
            load_c30 = LoadLowDegree()
            load_c30.configuration.set_file_id(c30_file_id)
            low_degs.update(load_c30.get_c30())

        rep = ReplaceLowDegree()
        rep.configuration.set_replace_deg1(degree1_or_not).set_replace_c20(c20_or_not).set_replace_c30(c30_or_not)
        rep.set_low_degrees(low_degs)

        shc = rep.apply_to(shc, dates[0], dates[1])

        '''deduct background'''
        if background is not None:
            shc.de_background(background=None if background == 'average' else background)

        self.times = ave_dates_GRACE
        self.shc_GRACE = shc

        return self

    def correct_gia(self):
        gia_model = self.configuration.get_GIA_model()
        lmax = self.configuration.get_lmax()

        load_gia = LoadGIA()
        load_gia.configuration.set_lmax(lmax)
        load_gia.configuration.set_GIA_model(gia_model)

        shc_gia_trend = load_gia.get_shc()

        gia = GIACorrectionSpectral()
        gia.configuration.set_times(self.times)
        gia.configuration.set_gia_trend(shc_gia_trend)

        self.shc_GRACE = gia.apply_to(self.shc_GRACE)

    def de_correlation(self):
        decorrelation_method = self.configuration.get_de_correlation_method()
        decorrelation_params = self.configuration.get_de_correlation_params()
        decorrelation_sliding_window_mode = self.configuration.get_de_correlation_sliding_window_mode()

        if self.filtered_shc is not None:
            shc_tobe_filtered = self.filtered_shc
        else:
            shc_tobe_filtered = self.shc_GRACE

        if decorrelation_method is None:
            shc_filtered = copy.deepcopy(shc_tobe_filtered)
        else:
            decorrelation = get_shc_decorrelation(decorrelation_method, decorrelation_params,
                                                  decorrelation_sliding_window_mode)
            shc_filtered = decorrelation.apply_to(shc_tobe_filtered)

        self.filtered_shc = shc_filtered

    def filter(self):
        shc_filter_method = self.configuration.get_shc_filter_method()
        shc_filter_params = self.configuration.get_shc_filter_params()
        lmax = self.configuration.get_lmax()

        shc_filter = get_shc_filter(shc_filter_method, shc_filter_params, lmax)

        if self.filtered_shc is not None:
            shc_filtered = shc_filter.apply_to(self.filtered_shc)

        else:
            shc_filtered = shc_filter.apply_to(self.shc_GRACE)

        self.filtered_shc = shc_filtered
        self.shc_filter = shc_filter

        return self

    def shc_to_grid(self, field_type=FieldPhysicalQuantity.EWH):
        # assert field_type == FieldPhysicalQuantity.EWH, "Only EWH is supported for now."

        lmax = self.configuration.get_lmax()

        if self.filtered_shc is not None:
            shc_tobe_processed = self.filtered_shc
        else:
            shc_tobe_processed = self.shc_GRACE

        '''convert shc quantity to ewh'''
        convert = ConvertSHC()
        convert.configuration.set_output_type(field_type)
        LN = LoveNumber()
        LN.configuration.set_lmax(lmax)
        ln = LN.get_Love_number()
        convert.set_Love_number(ln)
        shc = convert.apply_to(shc_tobe_processed)

        '''harmonic synthesis to grid'''
        har = self.harmonic
        special = field_type if field_type in (
            FieldPhysicalQuantity.HorizontalDisplacementEast,
            FieldPhysicalQuantity.HorizontalDisplacementNorth) else None
        grid = har.synthesis(shc, special_type=special)

        if self.filtered_shc is not None:
            self.filtered_grid = grid
        else:
            self.grid = grid

        return self

    def correct_leakage(self):
        leakage_method = self.configuration.get_leakage_method()
        leakage = get_leakage_deductor(leakage_method)

        if isinstance(leakage, ModelDriven):
            assert self.filtered_grid is not None
            assert self.harmonic is not None

            leakage.configuration.set_harmonic(self.harmonic)
            leakage.configuration.set_filter(self.shc_filter)

            '''load noah ewh'''
            # noah_model, model_times = get_TWS_series(self.configuration.get_begin_date(),
            #                                          self.configuration.get_end_date())

            noah_model, model_times = get_TWS_series(
                from_exist_results=FileTool.get_project_dir() / 'results/NOAH_EWH/NOAH_EWH_200204_202307.hdf5')

            leakage.configuration.set_model(noah_model)
            leakage.configuration.set_model_times(model_times)

            leakage.configuration.set_GRACE_times(self.times)

            for i in range(len(self.basin_map)):
                this_basin_map = self.basin_map[i]

                leakage.configuration.set_basin(this_basin_map)
                this_basin_acreage = MathTool.get_acreage(this_basin_map)

                # self.time_series_ewh = leakage.apply_to(self.filtered_grid) / this_basin_acreage
                time_series_value = leakage.apply_to(self.filtered_grid) / this_basin_acreage
                self.time_series_ewh.append(time_series_value)

        elif type(leakage) is BufferZone:
            leakage = BufferZone()

            leakage.configuration.set_harmonic(self.harmonic)
            leakage.configuration.set_filter(self.shc_filter)

            for i in range(len(self.basin_map)):
                this_basin_map = self.basin_map[i]

                leakage.configuration.set_basin(this_basin_map)
                this_basin_acreage = MathTool.get_acreage(this_basin_map)

                time_series_value = leakage.apply_to(self.filtered_grid) / this_basin_acreage
                self.time_series_ewh.append(time_series_value)

        else:
            print("Only Model driven and buffer-zone method is supported for now.")
            return -1

        return self

    def basin_average(self):
        if self.filtered_grid is not None:
            grid_tobe_processed = self.filtered_grid

        else:
            grid_tobe_processed = self.grid

        assert grid_tobe_processed is not None

        for i in range(len(self.basin_map)):
            this_basin_map = self.basin_map[i]

            this_basin_acreage = MathTool.get_acreage(this_basin_map)

            self.time_series_ewh.append(
                MathTool.global_integral(grid_tobe_processed.data * this_basin_map) / this_basin_acreage)

        return self

    def get_year_fraction(self):
        return TimeTool.convert_date_format(self.times,
                                            input_type=TimeTool.DateFormat.ClassDate,
                                            output_type=TimeTool.DateFormat.YearFraction)

    def get_ewh(self):
        if self.time_series_ewh is None:
            self.basin_average()

        return self.time_series_ewh


def demo_oldold():
    lmax = 96
    grid_space = 1

    shp_filepath_list = [
        FileTool.get_project_dir() / 'data/basin_shpfile/Danube_9_shapefiles',
        FileTool.get_project_dir() / 'data/basin_shpfile/Mississippi_10_shapefiles',
        FileTool.get_project_dir() / 'data/basin_shpfile/Brahmaputra_3_shapefiles',
        FileTool.get_project_dir() / 'data/basin_shpfile/MDB_4_shapefiles',
        FileTool.get_project_dir() / 'data/basin_shpfile/Rhine-Meuse_6_shapefiles'
    ]

    '''load GRACE L2 SH products'''
    load = LoadL2SH()
    load.configuration.set_begin_date(datetime.date(2002, 1, 1))
    load.configuration.set_end_date(datetime.date(2019, 3, 31))
    load.configuration.set_lmax(lmax)

    shc, dates = load.get_shc(with_dates=True)
    ave_dates_GRACE = TimeTool.get_average_dates(*dates)

    '''load and replace low degrees'''
    low_degs = {}

    load_deg1 = LoadLowDegree()
    load_deg1.configuration.set_file_id(L2LowDegreeFileID.TN13).set_institute(L2InstituteType.CSR)
    low_degs.update(load_deg1.get_results())

    load_c20 = LoadLowDegree()
    load_c20.configuration.set_file_id(L2LowDegreeFileID.TN14)
    low_degs.update(load_c20.get_results())

    rep = ReplaceLowDegree()
    rep.configuration.set_replace_c20().set_replace_deg1()
    rep.set_low_degrees(low_degs)

    shc = rep.apply_to(shc, dates[0], dates[1])

    '''deduct long-term average'''
    shc.de_background()

    '''load GIA and its correction'''
    load_gia = LoadGIA()
    load_gia.configuration.set_filepath(FileTool.get_project_dir() / 'data/GIA/GIA.ICE-6G_D.txt')
    load_gia.configuration.set_lmax(lmax)

    shc_gia_trend = load_gia.get_shc()

    gia = GIACorrectionSpectral()
    gia.configuration.set_times(ave_dates_GRACE)
    gia.configuration.set_gia_trend(shc_gia_trend)

    shc = gia.apply_to(shc)

    '''convert shc quantity to ewh'''
    convert = ConvertSHC()
    convert.configuration.set_output_type(FieldPhysicalQuantity.EWH)
    LN = LoveNumber()
    LN.configuration.set_lmax(lmax)
    ln = LN.get_Love_number()
    convert.set_Love_number(ln)
    shc = convert.apply_to(shc)

    '''filtering'''
    ddk3 = DDK()
    ddk3.configuration.set_filter_type(DDKFilterType.DDK3)

    shc_filtered = ddk3.apply_to(shc)

    '''harmonic synthesis to grid'''
    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)

    grids_filtered = har.synthesis(shc_filtered)

    '''leakage correction for each basin'''
    '''save results into a dict'''
    results_dict = dict(
        latitude=lat,
        longitude=lon,
        grids=grids_filtered.data,
        year_fraction=np.array(TimeTool.convert_date_format(ave_dates_GRACE,
                                                            input_type=TimeTool.DateFormat.ClassDate,
                                                            output_type=TimeTool.DateFormat.YearFraction
                                                            )),
        time=np.array(TimeTool.convert_date_format(ave_dates_GRACE,
                                                   input_type=TimeTool.DateFormat.ClassDate,
                                                   output_type=TimeTool.DateFormat.YMD
                                                   ))
    )

    # leak = Multiplicative()
    # leak = Scaling()
    # leak = Addictive()
    leak = ScalingGrid()

    leak.configuration.set_harmonic(har).set_filter(ddk3)

    noah_model, model_times = get_TWS_series(
        from_exist_results=FileTool.get_project_dir() / 'results/NOAH_EWH/NOAH_EWH.hdf5')

    leak.configuration.set_model(noah_model)
    leak.configuration.set_model_times(model_times)

    leak.configuration.set_GRACE_times(ave_dates_GRACE)
    leak.configuration.set_model_times(model_times)

    load_shp = ShpToMask()
    load_shp.configuration.set_grid_space(grid_space)
    for i in range(len(shp_filepath_list)):
        shp_filepath = shp_filepath_list[i]

        basin_name = shp_filepath.name.split('_')[0]
        print(basin_name)

        load_shp.configuration.set_shppath(shp_filepath)
        basin_maps_list = load_shp.get_basin_gridmap(with_whole=True)

        results_dict[basin_name] = {}
        results_dict[basin_name]['sub_basin'] = {}
        results_dict[basin_name]['scale_factor'] = {}
        results_dict[basin_name]['ewha'] = {}
        results_dict[basin_name]['area'] = {}

        for j in range(len(basin_maps_list)):
            basin_map = basin_maps_list[j]
            basin_shc = har.analysis(GRID(basin_map, har.lat, har.lon))
            basin_acreage = MathTool.get_acreage(basin_map)

            leak.configuration.set_basin(basin_shc)

            ewha = leak.apply_to(grids_filtered) / basin_acreage

            sub_basin_id = str(j)
            # scale_factor = leak.get_scale()

            results_dict[basin_name]['ewha'][sub_basin_id] = ewha
            results_dict[basin_name]['sub_basin'][sub_basin_id] = basin_map
            # results_dict[basin_name]['scale_factor'][sub_basin_id] = scale_factor
            results_dict[basin_name]['area'][sub_basin_id] = basin_acreage

    '''write file description and save as hdf5'''
    description = ("GRACE Data: CSR, RL06, max degree 96\n"
                   "Degree 1: replaced with auxiliary file TN-13 (CSR)\n"
                   "C20: replaced with auxiliary file TN-14\n"
                   "C30: replaced with auxiliary file TN-14\n"
                   "GIA: ICE-6G_D\n"
                   "Filter: DDK3\n"
                   "Leakage Correction: GriddedScaling, derived by GLDAS model.\n"
                   "Results attribute ewha is given in unit [m].\n"
                   "Results attribute area is given in unit [m^2].")

    results_dict['description'] = description

    with h5py.File(FileTool.get_project_dir() / 'results/ewh_20231101/NOAH_EWH_v3_gridded_scaling.hdf5', 'w') as f:
        dset_description = f.create_dataset("description", (1,), dtype=h5py.string_dtype())
        dset_description[0] = results_dict['description']

        # for key in ('latitude', 'longitude', 'grids', 'year_fraction'):
        # for key in ('latitude', 'longitude', 'time'):
        for key in ('latitude', 'longitude', 'year_fraction'):
            if key == 'time':
                f.create_dataset(key, data=results_dict[key])
                # f.create_dataset(key, data=results_dict[key], dtype=h5py.string_dtype())
            else:
                f.create_dataset(key, data=results_dict[key])

        for key in ('Danube', 'Mississippi', 'Rhine-Meuse', 'Brahmaputra', 'MDB'):
            this_group = f.create_group(key)

            for subkey in results_dict[key].keys():
                subkey_group = this_group.create_group(subkey)

                for subsubkey in results_dict[key][subkey].keys():
                    subkey_group.create_dataset(subsubkey, data=results_dict[key][subkey][subsubkey])


def post_processing_old(configuration: PostProcessingConfig):
    lmax = configuration.get_lmax()
    grid_space = configuration.get_grid_space()

    '''load basin SHC'''
    basin_name = configuration.get_basin().name
    basin_shc_filepath = FileTool.get_project_dir() / f'data/basin_mask/{basin_name}_maskSH.dat'
    basin_clm, basin_slm = load_SH_simple(basin_shc_filepath, key='', lmcs_in_queue=(1, 2, 3, 4), lmax=lmax)
    basin_shc = SHC(basin_clm, basin_slm)

    '''load GRACE L2 SH products'''
    begin_date, end_date = configuration.get_begin_date(), configuration.get_end_date()
    institute = configuration.get_GRACE_institute()
    load = LoadL2SH()

    load.configuration.set_begin_date(begin_date)
    load.configuration.set_end_date(end_date)
    load.configuration.set_institute(institute)
    load.configuration.set_lmax(lmax)

    shc, dates = load.get_shc(with_dates=True)
    ave_dates_GRACE = TimeTool.get_average_dates(*dates)

    '''load and replace low degrees'''
    degree1_or_not = 'degree1' in configuration.get_replace_low_degree_coefficients()
    degree1_file_id = configuration.get_degree1_file_id()

    c20_or_not = 'c20' in configuration.get_replace_low_degree_coefficients()
    c20_file_id = configuration.get_c20_file_id()

    c30_or_not = 'c30' in configuration.get_replace_low_degree_coefficients()
    c30_file_id = configuration.get_c30_file_id()

    low_degs = {}

    if degree1_or_not:
        load_deg1 = LoadLowDegree()
        load_deg1.configuration.set_file_id(degree1_file_id).set_institute(institute)
        low_degs.update(load_deg1.get_degree1())

    if c20_or_not:
        load_c20 = LoadLowDegree()
        load_c20.configuration.set_file_id(c20_file_id)
        low_degs.update(load_c20.get_c20())

    if c30_or_not:
        load_c30 = LoadLowDegree()
        load_c30.configuration.set_file_id(c30_file_id)
        low_degs.update(load_c30.get_c30())

    rep = ReplaceLowDegree()
    rep.configuration.set_replace_deg1(True).set_replace_c20(True).set_replace_c30(True)
    rep.set_low_degrees(low_degs)

    shc = rep.apply_to(shc, dates[0], dates[1])

    '''deduct long-term average'''
    shc.de_background()

    '''load GIA and its correction'''
    gia_model = configuration.get_GIA_model()

    load_gia = LoadGIA()
    load_gia.configuration.set_lmax(lmax)
    load_gia.configuration.set_GIA_model(gia_model)

    shc_gia_trend = load_gia.get_shc()

    gia = GIACorrectionSpectral()
    gia.configuration.set_times(ave_dates_GRACE)
    gia.configuration.set_gia_trend(shc_gia_trend)

    shc = gia.apply_to(shc)

    '''convert shc quantity to ewh'''
    convert = ConvertSHC()
    convert.configuration.set_output_type(FieldPhysicalQuantity.EWH)
    LN = LoveNumber()
    LN.configuration.set_lmax(lmax)
    ln = LN.get_Love_number()
    convert.set_Love_number(ln)
    shc = convert.apply_to(shc)

    '''de-correlation'''
    decorrelation_method = configuration.get_de_correlation_method()
    decorrelation_params = configuration.get_de_correlation_params()
    decorrelation_sliding_window_mode = configuration.get_de_correlation_sliding_window_mode()

    if decorrelation_method is None:
        shc_filtered = copy.deepcopy(shc)

    else:
        decorrelation = get_shc_decorrelation(decorrelation_method, decorrelation_params,
                                              decorrelation_sliding_window_mode)

        shc_filtered = decorrelation.apply_to(shc)

    '''filtering'''
    shc_filter_method = configuration.get_shc_filter_method()
    shc_filter_params = configuration.get_shc_filter_params()

    shc_filter = get_shc_filter(shc_filter_method, shc_filter_params, lmax)

    shc_filtered = shc_filter.apply_to(shc_filtered)

    '''harmonic synthesis to grid'''
    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)

    grids_filtered = har.synthesis(shc_filtered)

    '''leakage correction for each basin'''
    leakage_method = configuration.get_leakage_method()
    leakage = get_leakage_deductor(leakage_method)
    leakage.configuration.set_harmonic(har)
    leakage.configuration.set_filter(shc_filter)
    leakage.configuration.set_basin(basin_shc)

    if leakage_method in (LeakageMethod.Addictive, LeakageMethod.Multiplicative,
                          LeakageMethod.Scaling, LeakageMethod.ScalingGrid):
        '''load noah ewh'''
        noah_model, model_times = get_TWS_series(begin_date, end_date,
                                                 from_exist_results=FileTool.get_project_dir() / 'results/NOAH_EWH/NOAH_EWH.hdf5')

        leakage.configuration.set_model(noah_model)
        leakage.configuration.set_model_times(model_times)
        leakage.configuration.set_GRACE_times(ave_dates_GRACE)

    else:
        return -1

    basin_map = har.synthesis(basin_shc).data[0]
    basin_acreage = MathTool.get_acreage(basin_map)

    ewha_corrected = leakage.apply_to(grids_filtered) / basin_acreage

    return ave_dates_GRACE, ewha_corrected


def demo_old():
    config = PostProcessingConfig().set_from_json(
        FileTool.get_project_dir() / 'setting/post_processing/PostProcessing.json')

    t, ewha = post_processing_old(config)

    year_fraction = TimeTool.convert_date_format(t,
                                                 input_type=TimeTool.DateFormat.ClassDate,
                                                 output_type=TimeTool.DateFormat.YearFraction)

    plt.plot(year_fraction, ewha)
    plt.show()


def demo():
    sgea_figdata_path = FileTool.get_project_dir('demo/experimental/SGEA_paper/data/fig2_v2/')

    pp = PostProcessing()
    jsonpath = FileTool.get_project_dir() / 'setting/post_processing/PostProcessing.json'
    pp.configuration.set_from_json(jsonpath)

    pp.prepare()
    pp.load_files()  # load GRACE SHC, and replace low-degrees

    # load_gc = False  # if True, ignore pp.correct_gia, because geometric-corrected data had already been gia-corrected。
    # if True:
    #     cqlm, sqlm = [], []
    #     path = FileTool.get_project_dir('results/geometrical/CSR/replace_deg123/with_gia/')
    #     path_list = list(path.iterdir())
    #     path_list_to_load = []
    #     for i in range(len(path_list)):
    #         if ('not_corrected' in path_list[i].name) and not load_gc:
    #             path_list_to_load.append(path_list[i])
    #
    #         elif ('not_corrected' not in path_list[i].name) and load_gc:
    #             path_list_to_load.append(path_list[i])
    #
    #     for i in range(len(pp.times)):
    #         cslm = np.load(path_list_to_load[i])
    #         cqlm.append(cslm[0])
    #         sqlm.append(cslm[1])
    #
    #     pp.shc_GRACE = SHC(np.array(cqlm), np.array(sqlm))
    pp.shc_to_grid(field_type=FieldPhysicalQuantity.HorizontalDisplacementNorth)  # synthesis harmonic to (EWH) grid
    # pp.shc_to_grid(field_type=FieldPhysicalQuantity.EWH)  # synthesis harmonic to (EWH) grid

    # pp.correct_gia()  # GIA correction

    pp.de_correlation()  # de-correlation filter
    pp.filter()  # low-pass filter

    pp.shc_to_grid(field_type=FieldPhysicalQuantity.HorizontalDisplacementNorth)  # synthesis harmonic to (EWH) grid
    # pp.shc_to_grid(field_type=FieldPhysicalQuantity.EWH)  # synthesis harmonic to (EWH) grid

    # pp.correct_leakage()  # leakage correction
    # pp.basin_average()  # without leakage correction

    # times = pp.get_year_fraction()
    # values = pp.get_ewh()

    # np.save(sgea_figdata_path / 'gc_filtered_grids.npy', pp.filtered_grid.data)
    # np.save(sgea_figdata_path / 'filtered_grids.npy', pp.filtered_grid.data)

    # np.save(sgea_figdata_path / 'geoid_grids.npy', pp.grid.data)
    # np.save(sgea_figdata_path / 'filtered_geoid_grids.npy', pp.filtered_grid.data)

    # np.save(sgea_figdata_path / 'gravity_grids.npy', pp.grid.data)
    # np.save(sgea_figdata_path / 'filtered_gravity_grids.npy', pp.filtered_grid.data)

    # np.save(sgea_figdata_path / 'ewh_grids_gs300.npy', pp.grid.data)
    # np.save(sgea_figdata_path / 'filtered_ewh_grids_gs300.npy', pp.filtered_grid.data)

    np.save(sgea_figdata_path / 'displace_north_grids.npy', pp.grid.data)
    np.save(sgea_figdata_path / 'filtered_displace_north_grids.npy', pp.filtered_grid.data)

    # np.save(sgea_figdata_path / 'displace_east_grids.npy', pp.grid.data)
    # np.save(sgea_figdata_path / 'filtered_displace_east_grids.npy', pp.filtered_grid.data)

    # np.save(sgea_figdata_path / 'displace_vertical_grids.npy', pp.grid.data)
    # np.save(sgea_figdata_path / 'filtered_displace_vertical_grids.npy', pp.filtered_grid.data)

    # np.save(sgea_figdata_path / 'times.npy', TimeTool.convert_date_format(
    #     pp.times, input_type=TimeTool.DateFormat.ClassDate, output_type=TimeTool.DateFormat.YearFraction
    # ))
    #
    # np.save(sgea_figdata_path / 'ewh_Ocean_no_GIA.npy', pp.time_series_ewh)

    # config_dict = pp.configuration.__dict__
    # with open(FileTool.get_project_dir() / 'results/post_processing/res1/params.txt', 'w+') as f:
    #     for key in config_dict.keys():
    #         f.write(f'{key}\t{config_dict[key]}\n')

    # plt.plot(times, values[0])
    # plt.plot(times, values[1])
    # plt.plot(times, values[2])
    # plt.plot(times, values[3])
    # plt.plot(times, values[4])
    plt.show()


def demo2():
    sgea_figdata_path = FileTool.get_project_dir('demo/experimental/SGEA_paper/data/fig2_v2/')

    pp = PostProcessing()
    jsonpath = FileTool.get_project_dir() / 'setting/post_processing/PostProcessing.json'
    pp.configuration.set_from_json(jsonpath)

    pp.prepare()
    pp.load_files()  # load GRACE SHC, and replace low-degrees

    load_gaa = LoadL2SH()

    load_gaa.configuration.set_product_type(L2ProductType.GAA)
    load_gaa.configuration.set_begin_date(pp.configuration.get_begin_date())
    load_gaa.configuration.set_end_date(pp.configuration.get_end_date())
    load_gaa.configuration.set_institute(L2InstituteType.JPL)
    load_gaa.configuration.set_lmax(pp.configuration.get_lmax())

    pp.shc_GRACE, dates = load_gaa.get_shc(with_dates=True)
    pp.times = TimeTool.get_average_dates(*dates)

    pp.shc_GRACE.cs[:, 1:] = 0

    pp.shc_to_grid(field_type=FieldPhysicalQuantity.EWH)  # synthesis harmonic to (EWH) grid

    pp.correct_leakage()  # leakage correction

    pp.basin_average()  # without leakage correction

    times = pp.get_year_fraction()
    values = pp.get_ewh()

    # np.save(sgea_figdata_path / 'ewh_ocean_gmam.npy', pp.time_series_ewh)
    # np.save(sgea_figdata_path / 'gmam_times.npy', TimeTool.convert_date_format(
    #     pp.times, input_type=TimeTool.DateFormat.ClassDate, output_type=TimeTool.DateFormat.YearFraction
    # ))

    plt.plot(times, values[0])
    plt.show()


def demo3():
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib import rcParams
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cmaps
    import numpy as np
    from tqdm import trange

    from pysrc.auxiliary.tools.MathTool import MathTool

    sgea_figdata_path = FileTool.get_project_dir('demo/experimental/SGEA_paper/data/fig2_v2/')
    sgea_fig_path = FileTool.get_project_dir('demo/experimental/SGEA_paper/figs/')

    pp = PostProcessing()
    jsonpath = FileTool.get_project_dir() / 'setting/post_processing/PostProcessing.json'
    pp.configuration.set_from_json(jsonpath)
    pp.configuration.set_basin(BasinName.Amazon)
    pp.configuration.set_begin_date(datetime.date(2005, 1, 1))
    pp.configuration.set_end_date(datetime.date(2016, 1, 1))

    pp.prepare()
    pp.load_files()  # load GRACE SHC, and replace low-degrees

    # pp.shc_to_grid(field_type=FieldPhysicalQuantity.EWH)  # synthesis harmonic to (EWH) grid

    pp.correct_gia()  # GIA correction

    pp.de_correlation()  # de-correlation filter
    pp.filter()  # low-pass filter

    pp.shc_to_grid(field_type=FieldPhysicalQuantity.EWH)  # synthesis harmonic to (EWH) grid

    pp.correct_leakage()  # leakage correction
    # pp.basin_average()  # without leakage correction

    times = pp.get_year_fraction()
    values = pp.get_ewh()[0]

    # print(times)
    #
    # print(list(values))

    for i in range(len(times)):
        print(
            TimeTool.convert_date_format(times[i], input_type=TimeTool.DateFormat.YearFraction),
            ('' if values[i] < 0 else ' ') + '%.16f' % values[i],
            sep='\t'
        )

    # np.save(sgea_figdata_path / 'displace_north_grids.npy', pp.grid.data)
    # np.save(sgea_figdata_path / 'filtered_displace_north_grids.npy', pp.filtered_grid.data)

    fig = plt.figure(figsize=(3, 2.5))
    ax = fig.add_axes([0.25, 0.2, 0.73, 0.78])
    # ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
    #
    # lat, lon = MathTool.get_global_lat_lon_range(1)
    # lon2d, lat2d = np.meshgrid(lon, lat)
    #
    # norm = matplotlib.colors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
    # map_to_plot = pp.basin_map[0].copy()
    # map_to_plot[np.where(map_to_plot >= 0.5)] = 1
    # map_to_plot[np.where(map_to_plot < 1)] = 0
    #
    # ax.contourf(
    #     lon2d, lat2d, map_to_plot, 1,
    #     cmap=cmaps.hotcold_18lev_r,
    #     transform=ccrs.PlateCarree(),
    #     norm=norm
    # )
    #
    # ax.add_feature(cfeature.COASTLINE)
    #
    # latmin = -25
    # latmax = 10
    # lonmin = -85
    # lonmax = -45
    #
    # ax.set_extent([lonmin, lonmax, latmin, latmax])
    # ax.gridlines()
    #
    # x_extent = [-85, -65, -45]
    # y_extent = [-20, -5, 10]
    # ax.set_xticks(x_extent,
    #               # [str(np.abs(x_extent[i])) + ('$^\circ N$' if x_extent[i] > 0 else '$^\circ S$') for i in
    #               #  range(len(x_extent))],
    #               crs=ccrs.PlateCarree()
    #               )
    # ax.set_yticks(y_extent, crs=ccrs.PlateCarree())
    # ax.tick_params(color='blue', direction='in')
    #
    ax.set_ylabel('EWH (m)')

    ax.set_xlabel('Time (year)')
    ax.set_xticks([2005, 2010, 2015])
    ax.set_xlim(2005, 2016)

    ax.plot(times, values, color='black')

    plt.savefig(sgea_fig_path / 'ui_fig_pp_results_xy.pdf')
    plt.show()


if __name__ == '__main__':
    demo3()
