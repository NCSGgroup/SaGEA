import copy
import warnings

import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
import pysrc.auxiliary.preference.EnumClasses as Enums
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.auxiliary.preference.Constants import GeoConstants
from pysrc.auxiliary.preference.EnumClasses import match_string

from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
from pysrc.post_processing.filter.GetSHCFilter import get_filter
from pysrc.post_processing.geometric_correction.GeometricalCorrection import GeometricalCorrection
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.post_processing.replace_low_deg.ReplaceLowDegree import ReplaceLowDegree


class SHC:
    """
    This class is to store the spherical harmonic coefficients (SHCs) for the use in necessary data processing.

    Attribute self.value stores the coefficients in 2d-array (in numpy.ndarray) combined with c and s.
    which are sorted by degree, for example,
    numpy.ndarray: [[c1[0,0]; s1[1,1], c1[1,0], c1[1,1]; s1[2,2], s1[2,1], c1[2,0], c1[2,1], c1[2,2]; ...],
     [c2[0,0]; s2[1,1], c2[1,0], c2[1,1]; s2[2,2], s2[2,1], c2[2,0], c2[2,1], c2[2,2]; ...],
     [                                        ...                                         ]].
    Note that even it stores only one set of SHCs, the array is still 2-dimension, i.e.,
    [[c1[0,0]; s1[1,1], c1[1,0], c1[1,1]; s1[2,2], s1[2,1], c1[2,0], c1[2,1], c1[2,2]; ...]].

    Attribute self.dates stores beginning and ending dates (in datetime.date) in list as
        list: [
            [begin_1: datetime,date, begin_2: datetime,date, ...],
            [end_1: datetime,date, end_2: datetime,date, ...],
        ] if needed, else None.

    Attribute self.normalization indicates the normalization of the SHCs (in EnumClasses.SHNormalization), for example,
    EnumClasses.SHNormalization.full.

    Attribute self.physical_dimension indicates the physical dimension of the SHCs (in EnumClasses.PhysicalDimensions).
    """

    def __init__(self, c, s=None, dates=None, normalization=None, physical_dimension=None):
        """

        :param c: harmonic coefficients c in 2-dimension (l,m), or a series (q,l,m);
                if s is None:
                    1-dimension array sorted by degree [c00, s11, c10, c11, s22, s21, ...]
                    or 1-dimension array sorted by degree [[c00, s11, c10, c11, s22, s21, ...],[...]]
        :param s: harmonic coefficients s in 2-dimension (l,m), or a series (q,l,m),
                or None.
        :param dates: beginning and ending dates in list like
                [
                    [begin_1: datetime,date, begin_2: datetime,date, ..., begin_n: datetime,date],
                    [end_1: datetime,date, end_2: datetime,date, ..., end_n: datetime,date],
                ],
        :param normalization: in EnumClasses.SHNormalization, default EnumClasses.SHNormalization.full.
        :param physical_dimension: in EnumClasses.PhysicalDimensions, default EnumClasses.PhysicalDimensions.Dimensionless.
        """
        if s is None:
            self.value = np.array(c)
            if len(self.value.shape) == 1:
                self.value = np.array([self.value])

        else:
            assert np.shape(c) == np.shape(s)

            if len(np.shape(c)) == 2:
                self.value = MathTool.cs_combine_to_triangle_1d(c, s)

            elif len(np.shape(c)) == 3:
                cs = []
                for i in range(np.shape(c)[0]):
                    this_cs = MathTool.cs_combine_to_triangle_1d(c[i], s[i])
                    cs.append(this_cs)
                self.value = np.array(cs)

        if len(np.shape(self.value)) == 1:
            self.value = self.value[None, :]

        assert len(np.shape(self.value)) == 2

        self.dates = None

    def get_dates(self, astype: TimeTool.DateFormat = None):
        """
        return:
            if self.is_with_date() is True: tuple of
                (datetime.date of beginning dates, datetime.date of ending dates)
            else: None
        """

        if not self.is_with_date():
            warnings.warn("No date information in this SHC instance.")
            return None
        else:
            dates_begin, dates_end = self.dates

            if astype is None:
                return dates_begin, dates_end
            else:
                db = TimeTool.convert_date_format(
                    dates_begin,
                    input_type=TimeTool.DateFormat.ClassDate,
                    output_type=astype
                )
                de = TimeTool.convert_date_format(
                    dates_end,
                    input_type=TimeTool.DateFormat.ClassDate,
                    output_type=astype
                )

                return db, de

    def is_with_date(self):
        return self.dates is not None

    def append(self, *params, date_begin=None, date_end=None):
        """

        :param params: One parameter of instantiated SHC,
         or two parameters of c and s with the same input requirement as SHC.
        :param date_begin: datetime.date, necessary is self.is_with_date() is True.
        :param date_end: datetime.date, necessary is self.is_with_date() is True.
        :return: self
        """
        assert len(params) in (1, 2)
        if self.is_with_date():
            assert (date_begin is not None) and (date_end is not None)

        if len(params) == 1:
            if issubclass(type(params[0]), SHC):
                shc = params[0]
            else:
                shc = SHC(params[0])

        else:
            shc = SHC(*params)

        assert np.shape(shc.value)[-1] == np.shape(self.value)[-1]

        self.value = np.concatenate([self.value, shc.value])
        if self.is_with_date():
            self.dates[0].append(date_begin)
            self.dates[1].append(date_end)

        return self

    def is_series(self):
        """
        To determine whether the spherical harmonic coefficients stored in this class are one group or multiple groups.
        :return: bool, True if it stores multiple groups, False if it stores only one group.
        """
        return self.get_length() != 1

    def get_length(self):
        """
        To get the number of sets.
        """
        return np.shape(self.value)[0]

    def __len__(self):
        return self.get_length()

    def get_lmax(self):
        """

        :return: int, max degree/order of the spherical harmonic coefficients stored in this class.
        """
        length_of_cs1d = np.shape(self.value)[-1]
        lmax = int(np.sqrt(length_of_cs1d) - 1)

        return lmax

    def get_cs2d(self, fill_value=0):
        """
        return: cqlm, sqlm. Both cqlm and sqlm are 3-dimension, EVEN IF NOT self.is_series()
        """
        lmax = self.get_lmax()

        num_of_series = np.shape(self.value)[0]

        cqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))
        sqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))

        for i in range(num_of_series):
            this_cs = self.value[i]
            this_clm, this_slm = MathTool.cs_decompose_triangle1d_to_cs2d(this_cs)
            cqlm[i, :, :] = this_clm
            sqlm[i, :, :] = this_slm

        return cqlm, sqlm

    def __de_average(self):
        if self.is_series():
            self.value -= np.mean(self.value, axis=0)
        else:
            raise Exception

    def get_average(self):
        return SHC(np.mean(self.value, axis=0))

    def de_background(self, background=None):
        """
        if background is None, de average
        """
        if background is None:
            self.__de_average()

        else:
            assert isinstance(background, SHC)
            assert not background.is_series()

            self.value -= background.value

    @staticmethod
    def identity(lmax: int):
        basis_num = (lmax + 1) ** 2
        cs = np.eye(basis_num)

        return SHC(cs)

    def __add__(self, other):
        assert isinstance(other, SHC)
        assert self.get_lmax() == other.get_lmax()

        return SHC(self.value + other.value)

    def __sub__(self, other):
        assert isinstance(other, SHC)
        assert self.get_lmax() == other.get_lmax()

        return SHC(self.value - other.value)

    def add(self, shc, lbegin=None, lend=None):
        if (lend is None) and (lbegin is None):
            self.value += shc.value
            return self

        else:
            assert (lend is None) or (0 <= lend <= self.get_lmax())
            assert (lbegin is None) or (0 <= lbegin <= self.get_lmax())

            shc_copy = copy.deepcopy(shc)

            if lend is not None:
                shc_copy.value[:, (lend + 1) ** 2:] = 0
            if lbegin is not None:
                shc_copy.value[:, :lbegin ** 2] = 0

            return self.add(shc_copy)

    def subtract(self, shc, lbegin=None, lend=None):
        if (lend is None) and (lbegin is None):
            self.value -= shc.value
            return self

        else:
            assert (lend is None) or (0 <= lend <= self.get_lmax())
            assert (lbegin is None) or (0 <= lbegin <= self.get_lmax())

            shc_copy = copy.deepcopy(shc)

            if lend is not None:
                shc_copy.value[:, (lend + 1) ** 2:] = 0
            if lbegin is not None:
                shc_copy.value[:, :lbegin ** 2] = 0

            return self.subtract(shc_copy)

    def get_degree_rms(self):
        cqlm, sqlm = self.get_cs2d()
        return MathTool.get_degree_rms(cqlm, sqlm)

    def get_degree_rss(self):
        cqlm, sqlm = self.get_cs2d()
        return MathTool.get_degree_rss(cqlm, sqlm)

    def get_cumulative_degree_rss(self):
        cqlm, sqlm = self.get_cs2d()
        return MathTool.get_cumulative_rss(cqlm, sqlm)

    def get_std(self):
        cs_std = np.std(self.value, axis=0)
        return SHC(cs_std)

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

    def filter(self, method: Enums.SHCFilterType or Enums.SHCDecorrelationType, param: tuple = None):
        cqlm, sqlm = self.get_cs2d()
        filtering = get_filter(method, param, lmax=self.get_lmax())
        cqlm_f, sqlm_f = filtering.apply_to(cqlm, sqlm)
        self.value = MathTool.cs_combine_to_triangle_1d(cqlm_f, sqlm_f)

        return self

    def to_GRD(self, grid_space=None, special_type: Enums.PhysicalDimensions = None):
        from pysrc.data_class.GRD import GRD

        """pure synthesis"""

        if grid_space is None:
            grid_space = int(180 / self.get_lmax())
        assert special_type in (
            None,
            Enums.PhysicalDimensions.HorizontalDisplacementEast,
            Enums.PhysicalDimensions.HorizontalDisplacementNorth,
        )

        lat, lon = MathTool.get_global_lat_lon_range(grid_space)

        lmax = self.get_lmax()
        har = Harmonic(lat, lon, lmax, option=1)

        cqlm, sqlm = self.get_cs2d()
        grid_data = har.synthesis(cqlm, sqlm, special_type=special_type)
        grid = GRD(grid_data, lat, lon, option=1)

        grid.dates_series = self.dates

        return grid

    def synthesis(self, lat, lon, discrete: bool = False, special_type: Enums.PhysicalDimensions = None):
        """
        :param lat: numpy.ndarray, latitudes in unit degree
        :param lon: numpy.ndarray, longitudes in unit degree
        :param discrete: bool, if True, the params lat and lon represent each point, and should be of the same length;
            else params lat and lon represent the profiles. Default is False.
        :param special_type: enums.PhysicalDimensions, optional.
        """

        lmax = self.get_lmax()
        har = Harmonic(lat, lon, lmax, option=1, discrete=discrete)

        cqlm, sqlm = self.get_cs2d()
        grid_data = har.synthesis(cqlm, sqlm, special_type=special_type)

        return grid_data

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

    def regional_extraction(self, shc_region, normalize="4pi", average=True):
        assert normalize in ("4pi",)
        assert isinstance(shc_region, SHC)

        extraction = (shc_region.value @ self.value.T) * (GeoConstants.radius_earth ** 2)

        if average:
            extraction /= (shc_region.value[:, None, 0] * (GeoConstants.radius_earth ** 2))

        else:
            if normalize == "4pi":
                extraction *= (4 * np.pi)
            else:
                assert False

        return extraction
