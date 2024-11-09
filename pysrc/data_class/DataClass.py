import datetime
import pathlib

import h5py
import netCDF4
import numpy as np

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.core_data_class.CoreGRID import CoreGRID
from pysrc.auxiliary.core_data_class.CoreSHC import CoreSHC
import pysrc.auxiliary.preference.EnumClasses as Enums
from pysrc.auxiliary.preference.EnumClasses import match_string

from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
from pysrc.post_processing.filter.GetSHCFilter import get_filter
from pysrc.post_processing.geometric_correction.GeometricalCorrection import GeometricalCorrection
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.post_processing.leakage.Addictive import Addictive
from pysrc.post_processing.leakage.BufferZone import BufferZone
from pysrc.post_processing.leakage.DataDriven import DataDriven
from pysrc.post_processing.leakage.ForwardModeling import ForwardModeling
from pysrc.post_processing.leakage.Iterative import Iterative
from pysrc.post_processing.leakage.Multiplicative import Multiplicative
from pysrc.post_processing.leakage.Scaling import Scaling
from pysrc.post_processing.leakage.ScalingGrid import ScalingGrid
from pysrc.post_processing.replace_low_deg.ReplaceLowDegree import ReplaceLowDegree
from pysrc.post_processing.seismic_correction.SeismicCorrection import SeismicCorrection


class SHC(CoreSHC):
    def __init__(self, c, s=None):
        super().__init__(c, s)

    def __add__(self, other):
        assert issubclass(type(other), CoreSHC)

        return SHC(self.value + other.value)

    def __sub__(self, other):
        assert issubclass(type(other), CoreSHC)

        return SHC(self.value - other.value)

    def convert_type(self, from_type=None, to_type=None):
        types = list(Enums.PhysicalDimensions)
        types_string = [i.name.lower() for i in types]
        types += types_string

        if from_type is None:
            from_type = Enums.PhysicalDimensions.Dimensionless
        if to_type is None:
            to_type = Enums.PhysicalDimensions.Dimensionless

        assert (from_type.lower() if type(
            from_type) is str else from_type) in types, f"from_type must be one of {types}"
        assert (to_type.lower() if type(
            to_type) is str else to_type) in types, f"to_type must be one of {types}"

        if from_type is None:
            from_type = Enums.PhysicalDimensions.Dimensionless
        if to_type is None:
            to_type = Enums.PhysicalDimensions.Dimensionless

        if type(from_type) is str:
            from_type = match_string(from_type, Enums.PhysicalDimensions, ignore_case=True)
        if type(to_type) is str:
            to_type = match_string(to_type, Enums.PhysicalDimensions, ignore_case=True)
        lmax = self.get_lmax()
        LN = LoveNumber()
        LN.configuration.set_lmax(lmax)

        ln = LN.get_Love_number()

        convert = ConvertSHC()
        convert.configuration.set_Love_number(ln).set_input_type(from_type).set_output_type(to_type)

        self.value = convert.apply_to(self.value)
        return self

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

    def filter(self, method: Enums.SHCFilterType or Enums.SHCDecorrelationType, param: tuple = None):
        cqlm, sqlm = self.get_cs2d()
        filtering = get_filter(method, param, lmax=self.get_lmax())
        cqlm_f, sqlm_f = filtering.apply_to(cqlm, sqlm)
        self.value = MathTool.cs_combine_to_triangle_1d(cqlm_f, sqlm_f)

        return self

    def geometric(self, assumption: Enums.GeometricCorrectionAssumption, log=False):
        gc = GeometricalCorrection()
        cqlm, sqlm = self.get_cs2d()
        cqlm_new, sqlm_new = gc.apply_to(cqlm, sqlm, assumption=assumption, log=log)
        self.value = MathTool.cs_combine_to_triangle_1d(cqlm_new, sqlm_new)

        return self

    def replace_low_degs(self, dates_begin, dates_end, low_deg: dict,
                         deg1=True, c20=False, c30=False):
        assert len(dates_begin) == len(dates_end) == len(self.value)
        if deg1:
            c10, c11, s11 = True, True, True
        else:
            c10, c11, s11 = False, False, False
        replace_or_not = (c10, c11, s11, c20, c30)
        low_ids = ("c10", "c11", "s11", "c20", "c30")
        for i in range(len(low_ids)):
            if replace_or_not:
                assert low_ids[i] in low_deg.keys(), f"input low_deg should include key {low_ids[i]}"
        replace_low_degs = ReplaceLowDegree()
        replace_low_degs.configuration.set_replace_deg1(deg1).set_replace_c20(c20).set_replace_c30(c30)
        replace_low_degs.set_low_degrees(low_deg)
        cqlm, sqlm = replace_low_degs.apply_to(*self.get_cs2d(), begin_dates=dates_begin, end_dates=dates_end)
        self.value = MathTool.cs_combine_to_triangle_1d(cqlm, sqlm)

        return self

    def expand(self, time):
        assert not self.is_series()

        year_frac = TimeTool.convert_date_format(time,
                                                 input_type=TimeTool.DateFormat.ClassDate,
                                                 output_type=TimeTool.DateFormat.YearFraction)

        year_frac = np.array(year_frac)

        trend = self.value[0]
        value = year_frac[:, None] @ trend[None, :]
        return SHC(value)


class GRID(CoreGRID):
    def __init__(self, grid, lat, lon, option=1):
        super().__init__(grid, lat, lon, option)

    def to_SHC(self, lmax=None):
        grid_space = self.get_grid_space()

        if lmax is None:
            lmax = int(180 / grid_space)

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        har = Harmonic(lat, lon, lmax, option=1)

        grid_data = self.value
        cqlm, sqlm = har.analysis_for_gqij(grid_data)
        shc = SHC(cqlm, sqlm)

        return shc

    def leakage(self, method: str, basin: np.ndarray, filter_type: str, filter_params: tuple, lmax: int,
                # necessary params
                times=None, reference: dict = None,
                # extra params for model-driven methods
                prefilter_type: Enums.SHCFilterType = Enums.SHCFilterType.Gaussian, prefilter_params: tuple = (50,),
                # extra params for iterative
                scale_type: str = "trend", shc_unfiltered: SHC = None,
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

    def integral(self, mask=None, average=True):
        if average:
            assert mask is not None

        if isinstance(mask, CoreGRID):
            assert not mask.is_series()
            mask = mask.value[0]

        if mask is None:
            grids = self.value
        else:
            grids = self.value * mask

        lat, lon = self.lat, self.lon

        # grid_shape = np.shape(grids[0])
        #
        # if lat is None:
        #     lat = np.linspace(-90, 90, grid_shape[0])
        #
        # if lon is None:
        #     lon = np.linspace(-180, 180, grid_shape[1])
        #
        # colat_rad, lon_rad = MathTool.get_colat_lon_rad(lat, lon)
        #
        # dlat = np.abs(colat_rad[1] - colat_rad[0])
        # dlon = np.abs(lon_rad[1] - lon_rad[0])
        #
        # domega = np.sin(colat_rad) * dlat * dlon * radius_e ** 2
        #
        # # if for_square:
        # #     integral = np.einsum('pij,i->p', grids, domega ** 2)
        # # else:
        # #     integral = np.einsum('pij,i->p', grids, domega)
        # integral_result = np.einsum('pij,i->p', grids, domega)
        integral_result = MathTool.global_integral(grids, lat, lon)

        if average:
            integral_result /= MathTool.get_acreage(mask)

        return integral_result

    def limiter(self, threshold=0, beyond=1, below=0):
        index_beyond = np.where(self.value >= threshold)
        index_below = np.where(self.value < threshold)

        self.value[index_beyond] = beyond
        self.value[index_below] = below
        return self

    def savefile(self, filepath: pathlib.Path, filetype=None, rewrite=False, time_dim=None, description=None):
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


if __name__ == '__main__':
    pass
