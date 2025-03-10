import copy
import datetime
import pathlib
import warnings

import h5py
import netCDF4
import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
import pysrc.auxiliary.preference.EnumClasses as Enums

from pysrc.post_processing.de_aliasing.DeAliasing import DeAliasing
from pysrc.post_processing.filter.GetSHCFilter import get_filter
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.post_processing.leakage.Addictive import Addictive
from pysrc.post_processing.leakage.BufferZone import BufferZone
from pysrc.post_processing.leakage.DataDriven import DataDriven
from pysrc.post_processing.leakage.ForwardModeling import ForwardModeling
from pysrc.post_processing.leakage.Iterative import Iterative
from pysrc.post_processing.leakage.Multiplicative import Multiplicative
from pysrc.post_processing.leakage.Scaling import Scaling
from pysrc.post_processing.leakage.ScalingGrid import ScalingGrid
from pysrc.post_processing.seismic_correction.SeismicCorrection import SeismicCorrection


class GRD:
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

        if option == 0:
            self.lat = 90 - np.degrees(lat)
            self.lon = np.degrees(lon)

        else:
            self.lat = lat
            self.lon = lon

        pass

    def __add__(self, other):
        assert isinstance(other, GRD)
        assert other.lat == self.lat and other.lon == self.lon

        return GRD(self.value + other.value, lat=self.lat, lon=self.lon)

    def __sub__(self, other):
        assert isinstance(other, GRD)
        assert other.lat == self.lat and other.lon == self.lon

        return GRD(self.value - other.value, lat=self.lat, lon=self.lon)

    def append(self, grid, lat=None, lon=None, option=0):
        """

        :param grid: instantiated GRID or a 2d-array of index (lat, lon).
                        If 2d-array, the lat and lon range should be the same with self.lat and self.lon;
                        If instantiated GRID, params lat, lon and option are not needed.
        :param lat: co-latitude in [rad] if option=0 else latitude in [degree]
        :param lon: longitude in [rad] if option=0 else longitude in [degree]
        :param option:
        :return:
        """
        assert type(grid) in (GRD, np.ndarray)

        if type(grid) is GRD:
            assert lat is None and lon is None
            assert grid.lat == self.lat
            assert grid.lon == self.lon

        else:
            assert np.shape(grid)[-2:] == (len(self.lat), len(self.lon))
            grid = GRD(grid, self.lat, self.lon, option)

        array_to_append = grid.value if grid.is_series() else np.array([grid.value])
        array_self = self.value if self.is_series() else [self.value]

        self.value = np.concatenate([array_self, array_to_append])

        return self

    def is_series(self):
        """
        To determine whether the data stored in this class are one group or multiple groups.
        :return: bool, True if it stores multiple groups, False if it stores only one group.
        """
        return len(np.shape(self.value)) == 3

    def get_grid_space(self):
        """
        return: grid_space in unit [degree]
        """
        return round(self.lat[1] - self.lat[0], 2)

    def get_length(self):
        return self.value.shape[0]

    def to_SHC(self, lmax=None, special_type: Enums.PhysicalDimensions = None):
        from pysrc.data_class.SHC import SHC

        grid_space = self.get_grid_space()

        assert special_type in (
            None,
            Enums.PhysicalDimensions.HorizontalDisplacementEast,
            Enums.PhysicalDimensions.HorizontalDisplacementNorth,
        )

        if special_type in (
                Enums.PhysicalDimensions.HorizontalDisplacementEast,
                Enums.PhysicalDimensions.HorizontalDisplacementNorth):
            assert False, "Horizontal Displacement is not supported yet."

        if lmax is None:
            lmax = int(180 / grid_space)

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        har = Harmonic(lat, lon, lmax, option=1)

        grid_data = self.value
        cqlm, sqlm = har.analysis(grid_data, special_type=special_type)
        shc = SHC(cqlm, sqlm)

        return shc

    def analysis(self, lmax=None, from_type: Enums.PhysicalDimensions = None,
                 to_type: Enums.PhysicalDimensions = None):

        grid_copy = copy.deepcopy(self)

        shc = grid_copy.to_SHC(lmax=lmax)
        shc.convert_type(from_type=from_type, to_type=to_type)

        return shc

    def filter(self, method: Enums.GridFilterType, params: tuple = None):
        assert method in Enums.GridFilterType
        filtering = get_filter(method, params)
        self.value = filtering.apply_to(self.value, option=1)
        return self

    def leakage(self, method: Enums.LeakageMethod, basin: np.ndarray, filter_type, filter_params: tuple, lmax: int,
                # necessary params
                times=None, reference: dict = None,
                # extra params for model-driven methods
                prefilter_type: Enums.SHCFilterType = Enums.SHCFilterType.Gaussian, prefilter_params: tuple = (50,),
                # extra params for iterative
                scale_type: str = "trend", shc_unfiltered=None,
                # extra params for scaling and scaling_grid
                basin_conservation: np.ndarray = None, fm_iter_times: int = 30, log=False
                # extra params for forward modeling
                ):

        assert method in Enums.LeakageMethod
        methods_of_model_driven = (
            Enums.LeakageMethod.Addictive, Enums.LeakageMethod.Multiplicative,
            Enums.LeakageMethod.Scaling, Enums.LeakageMethod.ScalingGrid
        )

        methods_of_data_driven = (
            Enums.LeakageMethod.ForwardModeling, Enums.LeakageMethod.DataDriven,
            Enums.LeakageMethod.BufferZone, Enums.LeakageMethod.Iterative
        )

        grid_space = self.get_grid_space()
        lat, lon = MathTool.get_global_lat_lon_range(grid_space)
        har = Harmonic(lat, lon, lmax, option=1)

        if filter_type == Enums.GridFilterType.VGC and len(filter_params) <= 4:
            filter_params = list(filter_params) + [None] * (4 - len(filter_params)) + [har]
            filter_params = tuple(filter_params)

        filtering = get_filter(filter_type, filter_params, lmax=lmax)

        if method in methods_of_model_driven:
            assert {"time", "model"}.issubset(set(reference.keys()))

            if method == Enums.LeakageMethod.Addictive:
                lk = Addictive()

            elif method == Enums.LeakageMethod.Multiplicative:
                lk = Multiplicative()

            elif method == Enums.LeakageMethod.Scaling:
                lk = Scaling()

            elif method == Enums.LeakageMethod.ScalingGrid:
                lk = ScalingGrid()
                lk.configuration.set_scale_type(scale_type)

            else:
                assert False

            lk.configuration.set_GRACE_times(times)
            lk.configuration.set_model_times(reference["time"])

            if isinstance(reference["model"], GRD):
                reference["model"] = reference["model"].value
            lk.configuration.set_model(reference["model"])

        elif method in methods_of_data_driven:
            if method == Enums.LeakageMethod.DataDriven:
                assert shc_unfiltered is not None, "Data-driven requires parameter shc_unfiltered."

                lk = DataDriven()
                lk.configuration.set_cs_unfiltered(*shc_unfiltered.get_cs2d())

            elif method == Enums.LeakageMethod.BufferZone:
                lk = BufferZone()

            elif method == Enums.LeakageMethod.ForwardModeling:
                assert basin_conservation is not None, "Forward Modeling requires parameter basin_conservation."
                assert fm_iter_times is not None, "Forward Modeling requires parameter fm_iter_times."

                lk = ForwardModeling()
                lk.configuration.set_basin_conservation(basin_conservation)
                lk.configuration.set_max_iteration(fm_iter_times)
                lk.configuration.set_print_log(log)

            elif method == Enums.LeakageMethod.Iterative:
                assert (prefilter_params is not None) and (
                        prefilter_type is not None), "Iterative requires parameter prefilter_type and prefilter_params."
                assert shc_unfiltered is not None, "Iterative requires parameter shc_unfiltered."

                lk = Iterative()

                prefilter = get_filter(prefilter_type, prefilter_params, lmax=lmax)
                lk.configuration.set_prefilter(prefilter)
                lk.configuration.set_cs_unfiltered(*shc_unfiltered.get_cs2d())

            else:
                assert False

        else:
            assert False

        lk.configuration.set_basin(basin)
        lk.configuration.set_filter(filtering)
        lk.configuration.set_harmonic(har)

        gqij_corrected = lk.apply_to(self.value, get_grid=True)
        self.value = gqij_corrected

        return self

    def seismic(self, dates, events: pathlib.Path = None):
        if events is None:
            events = FileTool.get_project_dir('setting/post_processing/earthquakes.json')

        sei = SeismicCorrection()
        sei.configuration.set_times(dates)
        sei.configuration.set_earthquakes(events)

        sei.apply_to(self.value, lat=self.lat, lon=self.lon)

        return self

    def de_aliasing(self, dates,
                    s2: bool = False, p1: bool = False, s1: bool = False, k2: bool = False, k1: bool = False):
        de_alias = DeAliasing()

        de_alias.configuration.set_de_s2(s2),
        de_alias.configuration.set_de_p1(p1),
        de_alias.configuration.set_de_s1(s1),
        de_alias.configuration.set_de_k2(k2),
        de_alias.configuration.set_de_k1(k1),

        year_frac = TimeTool.convert_date_format(
            dates, input_type=TimeTool.DateFormat.ClassDate, output_type=TimeTool.DateFormat.YearFraction
        )

        self.value = de_alias.apply_to(self.value, year_frac)

    def __integral_for_one_basin(self, mask=None, average=True):
        assert type(mask) in (np.ndarray,) or isinstance(mask, GRD) or mask is None

        if average:
            assert mask is not None

        if isinstance(mask, GRD):
            assert not mask.is_series()
            mask = mask.value[0]

        if mask is None:
            grids = self.value
        else:
            grids = self.value * mask

        lat, lon = self.lat, self.lon

        integral_result = MathTool.global_integral(grids, lat, lon)

        if average:
            integral_result /= MathTool.get_acreage(mask)

        return integral_result

    def integral(self, mask=None, average=True):
        assert type(mask) in (np.ndarray,) or isinstance(mask, GRD) or mask is None

        if mask is None:
            return self.__integral_for_one_basin(mask, average=average)

        else:
            if isinstance(mask, GRD):
                mask_value = mask.value
            elif type(mask) in (np.ndarray,):
                mask_value = mask
            else:
                assert False

            assert mask_value.ndim in (2, 3)

            if mask_value.ndim == 2:
                return self.__integral_for_one_basin(mask, average=average)

            else:
                result_list = []
                for i in range(mask_value.shape[0]):
                    result = self.__integral_for_one_basin(mask_value[i], average=average)
                    result_list.append(result)

                return np.array(result_list)

    def regional_extraction(self, grid_region, average=True):
        assert isinstance(grid_region, GRD)

        return self.integral(grid_region.value, average=average)

    def limiter(self, threshold=0, beyond=1, below=0):
        index_beyond = np.where(self.value >= threshold)
        index_below = np.where(self.value < threshold)

        self.value[index_beyond] = beyond
        self.value[index_below] = below
        return self

    def savefile(self, filepath: pathlib.Path, filetype=None, rewrite=False, time_dim=None, description=None):
        warnings.warn("This method will be removed in future versions, use to_file() instead", DeprecationWarning)
        self.to_file(filepath, filetype=filetype, rewrite=rewrite, time_dim=time_dim, description=description)

    def to_file(self, filepath: pathlib.Path, filetype=None, rewrite=False, time_dim=None, description=None):
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        if filepath.exists() and not rewrite:
            assert False, "file already exists"

        filename = filepath.name
        if "." in filename:
            type_in_name = filename.split(".")[-1]
        else:
            type_in_name = None

        if type_in_name is None:
            if filetype is not None:
                filename += "." + filetype
            else:
                filename += ".nc"

        elif (type_in_name is not None) and (filetype is not None) and (type_in_name != filetype):
            filename += "." + filetype

        savetype = filename.split(".")[-1]

        types = ("nc", "npz", "hdf5")
        assert savetype in types, f"saving type must be one of {types}"

        if savetype == "nc":
            self.__save_nc(filepath, time_dim=time_dim, value_description=description)

        elif savetype == "npz":
            self.__save_npz(filepath, time_dim=time_dim, value_description=description)

        elif savetype == "hdf5":
            self.__save_hdf5(filepath, time_dim=time_dim, value_description=description)

    def __save_nc(self, filepath: pathlib.Path, time_dim=None, from_date=None, value_description=None):
        assert filepath.name.endswith(".nc")
        assert time_dim is not None
        assert len(time_dim) == np.shape(self.value)[0]

        if from_date is None:
            from_date = datetime.date(1900, 1, 1)

        time_delta = TimeTool.convert_date_format(
            time_dim,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.TimeDelta,
            from_date=from_date,
        )

        with netCDF4.Dataset(filepath, 'w', format='NETCDF4') as ncfile:
            ncfile.createDimension('time', size=len(time_delta))
            ncfile.createDimension('lat', size=len(self.lat))
            ncfile.createDimension('lon', size=len(self.lon))

            times = ncfile.createVariable('time', int, ('time',))
            latitudes = ncfile.createVariable('lat', np.float32, ('lat',))
            longitudes = ncfile.createVariable('lon', np.float32, ('lon',))
            values = ncfile.createVariable('value', np.float32, ('time', 'lat', 'lon'))

            times[:] = time_delta
            latitudes[:] = self.lat
            longitudes[:] = self.lon
            values[:] = self.value

            times.description = f"days from {from_date}"
            latitudes.description = f"geographical latitude in unit [degree]"
            longitudes.description = f"geographical longitude in unit [degree]"
            if value_description is not None:
                values.description = value_description

        return self

    def __save_npz(self, filepath: pathlib.Path, time_dim=None, from_date=None, value_description=None):
        assert filepath.name.endswith(".npz")
        assert time_dim is not None
        assert len(time_dim) == np.shape(self.value)[0]

        if from_date is None:
            from_date = datetime.date(1900, 1, 1)

        time_delta = TimeTool.convert_date_format(
            time_dim,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.TimeDelta,
            from_date=from_date,
        )

        np.savez(
            filepath,
            lat=self.lat, lon=self.lon, value=self.value,
            description=value_description,
            date_begin=from_date, days=time_delta,
        )

    def __save_hdf5(self, filepath: pathlib.Path, time_dim=None, from_date=None, value_description=None):
        assert filepath.name.endswith(".hdf5")
        assert time_dim is not None
        assert len(time_dim) == np.shape(self.value)[0]

        if from_date is None:
            from_date = datetime.date(1900, 1, 1)

        time_delta = TimeTool.convert_date_format(
            time_dim,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.TimeDelta,
            from_date=from_date,
        )

        with h5py.File(filepath, "w") as h5file:
            t_group = h5file.create_group("time")
            t_group.create_dataset("description", data=f"days from {from_date}")
            t_group.create_dataset("data", data=time_delta)

            v_group = h5file.create_group("value")
            if value_description is not None:
                v_group.create_dataset("description", data=value_description)
            v_group.create_dataset("data", data=self.value)

            lat_group = h5file.create_group("lat")
            lat_group.create_dataset("description", data=f"geographical latitude in unit [degree]")
            lat_group.create_dataset("data", data=self.lat)

            lon_group = h5file.create_group("lon")
            lon_group.create_dataset("description", data=f"geographical longitude in unit [degree]")
            lon_group.create_dataset("data", data=self.lon)
