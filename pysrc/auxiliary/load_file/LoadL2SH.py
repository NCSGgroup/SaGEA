import re
import datetime
import json
from pathlib import Path, WindowsPath

import numpy as np

from pysrc.auxiliary.preference.EnumClasses import L2ProductType, L2InstituteType, L2Release
from pysrc.auxiliary.scripts.MatchConfigWithEnums import match_config
from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool
from pysrc.data_class.DataClass import SHC


def load_SHC(*filepath, key: str, lmax: int, lmcs_in_queue=None):
    """

    :param filepath: path of SH file
    :param key: '' if there is not any key.
    :param lmax: max degree and order.
    :param lmcs_in_queue: iter, Number of columns where degree l, order m, coefficient clm, and slm are located.
    :return: 2d tuple, whose elements are clm and slm in form of 2d array.
    """
    if len(filepath) == 1:
        def are_all_num(x: list):
            for i in lmcs_in_queue:
                if x[i - 1].replace('e', '').replace('E', '').replace('E', '').replace('E', '').replace('-',
                                                                                                        '').replace(
                    '+', '').replace('.', '').isnumeric():
                    pass
                else:
                    return False

            return True

        if lmcs_in_queue is None:
            lmcs_in_queue = [2, 3, 4, 5]

        l_queue = lmcs_in_queue[0]
        m_queue = lmcs_in_queue[1]
        c_queue = lmcs_in_queue[2]
        s_queue = lmcs_in_queue[3]

        mat_shape = (lmax + 1, lmax + 1)
        clm, slm = np.zeros(mat_shape), np.zeros(mat_shape)

        with open(filepath[0]) as f:
            txt_list = f.readlines()
            for i in range(len(txt_list)):
                if txt_list[i].startswith(key):
                    this_line = txt_list[i].split()

                    # if len(this_line) == 4 and are_all_num(this_line):
                    if are_all_num(this_line):
                        l = int(this_line[l_queue - 1])
                        if l > lmax:
                            continue

                        m = int(this_line[m_queue - 1])

                        clm[l, m] = float(this_line[c_queue - 1])
                        slm[l, m] = float(this_line[s_queue - 1])

                    else:
                        continue

        return clm, slm

    else:
        cqlm, sqlm = [], []
        for i in range(len(filepath)):
            clm, slm = load_SHC(filepath[i], key=key, lmax=lmax, lmcs_in_queue=lmcs_in_queue)

            cqlm.append(clm)
            sqlm.append(slm)

        return np.array(cqlm), np.array(sqlm)


class LoadL2SHSingleFile:
    """
    This class is to load one GRACE Level2 product.
    self.results is supposed to be a dict, including:
        'clm': np.ndarray in shape (lmax+1, lmax+1), .[l,m] == 0 if l < m;
        'slm': np.ndarray in shape (lmax+1, lmax+1), .[l,m] == 0 if l <= m;
        'begin_date': datetime.date;
        'end_date': datetime.date;
        'unused_dates': list of datetime.date on which satellite data was not used in solution.
    Also, each param can be called respectively, like self.clm2d
    """

    def __init__(self, filepath: Path, lmax: int = None):
        self.filepath = filepath

        self.results = {}
        self.clm = None
        self.slm = None
        self.begin_date = None
        self.end_date = None
        self.unused_dates = None
        self.lmax = lmax

        self.sigma_clm = None
        self.sigma_slm = None

        self._load()
        pass

    def _load(self):

        with open(self.filepath) as f:
            txt = f.read()

        '''load beginning and ending dates, and unused days.'''
        filename = self.filepath.name

        if 'UTCSR' in filename or 'GFZOP' in filename or 'JPLEM' in filename:
            pat_key = '(GSM|GAA|GAB|GAC|GAD)-2_([0-9]{7})-([0-9]{7})'
            groups = re.search(pat_key, filename, re.M).groups()  # ('GSM', '2008001', '2008031')

            begin_year_first = datetime.date(int(groups[1][:4]), 1, 1)
            begin_days = int(groups[1][4:])
            begin_date = begin_year_first + datetime.timedelta(begin_days - 1)

            end_year_first = datetime.date(int(groups[2][:4]), 1, 1)
            end_days = int(groups[2][4:])
            end_date = end_year_first + datetime.timedelta(end_days - 1)

            pat_unused_days = r'unused_days.*(\[.*\])'
            unues = re.search(pat_unused_days, txt)
            unues_dates = []
            if unues is not None:
                pat_unused_days_each = r'\d{4}-\d{2}-\d{2}'
                unues_each = re.findall(pat_unused_days_each, unues.group())
                for i in range(len(unues_each)):
                    ymd = unues_each[i].split('-')
                    year, month, day = int(ymd[0]), int(ymd[1]), int(ymd[2])
                    unues_dates.append(datetime.date(year, month, day))

        elif 'ITSG' in filename:
            year_month_pattern = r'n\d{2}_(\d{4})-(\d{2})'
            year, month = re.search(year_month_pattern, filename).groups()
            begin_date = datetime.date(int(year), int(month), 1)
            end_date = TimeTool.get_the_final_day_of_this_month(begin_date)

            unues_dates = []

        else:
            assert False

        '''load Clm2d, Slm2d'''
        pat_degree = r'degree *:? *\d+'
        deg_of_this_file = int(re.search(pat_degree, txt).group().split()[-1])

        if self.lmax is not None and deg_of_this_file < self.lmax:
            self.lmax = deg_of_this_file

        clm = np.zeros((deg_of_this_file + 1, deg_of_this_file + 1))
        slm = np.zeros((deg_of_this_file + 1, deg_of_this_file + 1))

        sigma_clm = np.zeros((deg_of_this_file + 1, deg_of_this_file + 1))
        sigma_slm = np.zeros((deg_of_this_file + 1, deg_of_this_file + 1))

        key1 = 'gfc'
        key2 = 'GRCOF2'
        pat_data = key1 + r'.*\d|' + key2 + r'.*\d'
        data = re.findall(pat_data, txt)
        for i in data:
            line = i.split()
            l = int(line[1])

            if self.lmax is not None and l >= self.lmax:
                continue

            m = int(line[2])
            clm[l][m] = float(line[3])
            slm[l][m] = float(line[4])

            sigma_clm[l][m] = float(line[5])
            sigma_slm[l][m] = float(line[6])

        if self.lmax is not None and self.lmax < deg_of_this_file:
            clm = clm[:self.lmax + 1, :self.lmax + 1]
            slm = slm[:self.lmax + 1, :self.lmax + 1]

        self.lmax = deg_of_this_file

        self.results.update(dict(clm=clm,
                                 slm=slm,
                                 sigma_clm=sigma_clm,
                                 sigma_slm=sigma_slm,
                                 begin_date=begin_date,
                                 end_date=end_date,
                                 unused_dates=unues_dates))

        self.clm = clm
        self.slm = slm
        self.begin_date = begin_date
        self.end_date = end_date
        self.unused_dates = unues_dates
        self.sigma_clm = sigma_clm
        self.sigma_slm = sigma_slm

        return self

    def get_shc(self):
        return SHC(self.clm, self.slm)

    def get_sigma_shc(self):
        return SHC(self.sigma_clm, self.sigma_slm)

    def get_diag_matrix(self):
        """
        return variances in the form of a diagonal matrix whose non-diagonal elements are set zero.
        sorted by degree, like
        diag(var(c00), var(c10), var(c11), var(s11), var(c20), var(c21), var(s21), var(c22), var(s22), ...)
        :return: 2d-array, in shape ((lmax+1)**2, (lmax+1)**2).
        """
        ones_tril = np.tril(np.ones((self.lmax + 1, self.lmax + 1)))
        ones_tri = np.concatenate([ones_tril[:, -1:0:-1], ones_tril], axis=1)

        sigma_cs_tri = np.concatenate([self.sigma_slm[:, -1:0:-1], self.sigma_clm], axis=1)
        sigma_cs_tri_fla = sigma_cs_tri[np.where(ones_tri == 1)].flatten()

        return np.diag(sigma_cs_tri_fla ** 2)


class LoadL2SHConfig:
    def __init__(self):
        """default setting"""
        self.product_type = L2ProductType.GSM
        self.institute = L2InstituteType.CSR
        self.release = L2Release.RL06
        self.lmax = 60
        self.beginning_date = datetime.date(2002, 4, 1)
        self.ending_date = datetime.date(2002, 4, 28)

    def set_product_type(self, product_type: L2ProductType):
        self.product_type = product_type
        return self

    def set_institute(self, institute: L2InstituteType):
        self.institute = institute
        return self

    def set_release(self, release: L2Release):
        self.release = release
        return self

    def set_begin_date(self, beginning_date: datetime.date or str):
        self._set_date(beginning_date, 'beginning')
        return self

    def set_end_date(self, ending_date: datetime.date or str):
        self._set_date(ending_date, 'ending')
        return self

    def _set_date(self, date, beginning_or_ending):
        assert type(date) in (datetime.date, str)
        assert beginning_or_ending in ('beginning', 'ending')

        if type(date) is datetime.date:
            if beginning_or_ending == 'beginning':
                self.beginning_date = date
            else:
                self.ending_date = date
        else:
            ymd = date.split('-')
            ymd = tuple((int(ymd[i]) for i in range(len(ymd))))  # year, month, day in int
            if beginning_or_ending == 'beginning':
                self.beginning_date = datetime.date(*ymd)
            else:
                self.ending_date = datetime.date(*ymd)

    def set_from_json(self, path: Path):
        """
        This function is to set or update parameters through external JSON files. The JSON file must contain these
            variables: "product_type", "institute" and "release", "lmax", "beginning_date" and "ending_date".
        :param path: a standardized class of pathlib.Path, and the filename should end with ".json"
        :return:
        """

        # ensure that the path describes a JSON file
        assert type(path) is WindowsPath and path.name.endswith('.json')

        # load json file to a dict
        with open(path) as jsonfile:
            json_dict = json.load(jsonfile)

        # ensure that the necessary parameters are included in the JSON file
        assert {'product_type', 'institute', 'release', 'beginning_date', 'ending_date', } <= set(json_dict.keys())

        # transmitting parameters for enum classes
        json_keys = ('product_type', 'institute', 'release')
        enum_classes = (L2ProductType, L2InstituteType, L2Release)
        setting_functions = (self.set_product_type, self.set_institute, self.set_release)

        match_config(json_dict, json_keys, enum_classes, setting_functions)

        # transmitting parameters for not enum classes
        self.set_begin_date(json_dict['beginning_date'])
        self.set_end_date(json_dict['ending_date'])

        return self

    def set_lmax(self, lmax):
        self.lmax = lmax

        return self


class LoadL2SH:
    """
    This class is to load GRACE Level2 products.
    """

    def __init__(self):
        self.configuration = LoadL2SHConfig()
        pass

    def config(self, config: LoadL2SHConfig):
        self.configuration = config
        return self

    def get_shc(self, with_dates: bool = False, get_sigma=False):
        """
        Obtain shc, or simultaneously obtain the corresponding dates.
        If dates is obtained, a tuple will be returned, where the 1st element is shc,
        and the 2nd element is a tuple composed of corresponding start date, end date, and unused dates.
        :param with_dates: bool, if True, return shc, (begin dates, end dates and unused dates); else return only shc.
        """
        filepath_list = self._get_filepath_list_to_load()

        '''load for each filepath in filepath list'''
        shc = None

        dates_begin = []
        dates_end = []
        dates_unused = []

        for i in range(len(filepath_list)):
            load = LoadL2SHSingleFile(filepath_list[i], lmax=self.configuration.lmax)
            if get_sigma:
                this_shc = load.get_sigma_shc()
            else:
                this_shc = load.get_shc()

            if shc is None:
                shc = this_shc
            else:
                shc.append(this_shc)

            dates_begin.append(load.begin_date)
            dates_end.append(load.end_date)
            dates_unused.append(load.unused_dates)

        if with_dates:
            return shc, (dates_begin, dates_end, dates_unused)

        else:
            return shc

    def _get_filepath_list_to_load(self):
        product_type = self.configuration.product_type
        institute = self.configuration.institute
        release = self.configuration.release
        lmax = self.configuration.lmax
        begin_date = self.configuration.beginning_date
        end_date = self.configuration.ending_date

        '''get filepath list to load'''
        filepath_list = []
        for year in range(begin_date.year, end_date.year + 1):
            dir_up_to_year = FileTool.get_l2_SH_dir_upto_year(year, product_type, institute, release, lmax)

            filelist_this_year = list(Path.iterdir(dir_up_to_year))

            if begin_date.year < year < end_date.year:
                filepath_list += filelist_this_year

            else:
                for i in range(len(filelist_this_year)):
                    this_filepath = filelist_this_year[i]
                    this_filename = this_filepath.name

                    if self.configuration.institute in (L2InstituteType.CSR, L2InstituteType.GFZ, L2InstituteType.JPL):
                        year_day_pattern = r'[A-Z]{3}-2_(\d{7})-(\d{7})'
                        beginning_year_day, ending_year_day = re.search(year_day_pattern, this_filename).groups()

                        this_begin_date = TimeTool.convert_date_format(beginning_year_day,
                                                                       input_type=TimeTool.DateFormat.YearDay,
                                                                       output_type=TimeTool.DateFormat.ClassDate)

                        this_end_date = TimeTool.convert_date_format(ending_year_day,
                                                                     input_type=TimeTool.DateFormat.YearDay,
                                                                     output_type=TimeTool.DateFormat.ClassDate)

                        this_ave_date = this_begin_date + (this_end_date - this_begin_date) / 2

                    elif self.configuration.institute == L2InstituteType.ITSG:
                        year_month_pattern = r'n\d{2}_(\d{4})-(\d{2})'
                        year, month = re.search(year_month_pattern, this_filename).groups()
                        this_begin_date = datetime.date(int(year), int(month), 1)
                        this_end_date = TimeTool.get_the_final_day_of_this_month(this_begin_date)

                        this_ave_date = this_begin_date + (this_end_date - this_begin_date) / 2

                    else:
                        assert False

                    if begin_date <= this_ave_date <= end_date:
                        filepath_list.append(this_filepath)

        return filepath_list


def demo():
    from pysrc.auxiliary.scripts.PlotGrids import plot_grids
    from pysrc.post_processing.geometric_correction.GeometricalCorrection import GeometricalCorrection

    load = LoadL2SH()
    load.configuration.set_institute(L2InstituteType.ITSG)
    load.configuration.set_release(L2Release.ITSGGrace2018)
    load.configuration.set_lmax(60)
    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2005, 12, 31))
    shc_ITSG, dates_ITSG = load.get_shc(with_dates=True)
    shc_ITSG.de_background()

    # load.configuration.set_institute(L2InstituteType.CSR)
    # load.configuration.set_release(L2Release.RL06)
    # load.configuration.set_lmax(60)
    # shc_CSR, dates_CSR = load.get_shc(with_dates=True)
    # shc_CSR.de_background()

    gc = GeometricalCorrection()
    shc_ITSG_gc = gc.apply_to(shc_ITSG)

    grid_ITSG = shc_ITSG.to_grid(grid_space=0.5)
    grid_ITSG_gc = shc_ITSG_gc.to_grid(grid_space=0.5)

    index = 0
    plot_grids(
        np.array([
            grid_ITSG.data[index],
            grid_ITSG_gc.data[index],
            grid_ITSG.data[index] - grid_ITSG_gc.data[index]
        ]),
        grid_ITSG.lat, grid_ITSG.lon,
        # [None, None, -0.1], [None, None, 0.1]
        [None, None, -0.1], [None, None, 0.1]
    )


if __name__ == '__main__':
    demo()
