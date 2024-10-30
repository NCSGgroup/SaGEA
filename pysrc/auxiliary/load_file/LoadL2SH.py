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


def match_dates_from_filename(filename):
    match_flag = False
    this_date_begin, this_date_end = None, None

    '''date format: yyyymmdd-yyyymmdd or yyyy-mm-dd-yyyy-mm-dd'''
    if not match_flag:
        date_begin_end_pattern = r"(\d{4})-?(\d{2})-?(\d{2})-(\d{4})-?(\d{2})-?(\d{2})"
        date_begin_end_searched = re.search(date_begin_end_pattern, filename)

        if date_begin_end_searched is not None:
            date_begin_end = date_begin_end_searched.groups()
            this_date_begin = datetime.date(*list(map(int, date_begin_end[:3])))
            this_date_end = datetime.date(*list(map(int, date_begin_end[3:])))

            match_flag = True

    '''date format: yyyyddd-yyyyddd'''
    if not match_flag:
        date_begin_end_pattern = r"(\d{4})(\d{3})-(\d{4})(\d{3})"
        date_begin_end_searched = re.search(date_begin_end_pattern, filename)

        if date_begin_end_searched is not None:
            date_begin_end = date_begin_end_searched.groups()
            this_date_begin = datetime.date(int(date_begin_end[0]), 1, 1) + datetime.timedelta(
                days=int(date_begin_end[1]) - 1)
            this_date_end = datetime.date(int(date_begin_end[2]), 1, 1) + datetime.timedelta(
                days=int(date_begin_end[3]) - 1)

            match_flag = True

    '''date format: yyyymm'''
    if not match_flag:
        date_begin_end_pattern = r"_(\d{4})(\d{2})_"
        date_begin_end_searched = re.search(date_begin_end_pattern, filename)

        if date_begin_end_searched is not None:
            year_month = date_begin_end_searched.groups()
            year = int(year_month[0])
            month = int(year_month[1])

            this_date_begin = datetime.date(int(year), month, 1)
            if month < 12:
                this_date_end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
            elif month == 12:
                this_date_end = datetime.date(year + 1, 1, 1) + datetime.timedelta(days=1)
            else:
                assert False

            match_flag = True

    assert match_flag, f"illegal date format in filename: {filename}"

    return this_date_begin, this_date_end


def load_SHC(*filepath, key: str, lmax: int, lmcs_in_queue=None, get_dates=False, begin_date=None, end_date=None):
    """

    :param filepath: path of SH file
    :param key: '' if there is not any key.
    :param lmax: max degree and order.
    :param lmcs_in_queue: iter, Number of columns where degree l, order m, coefficient clm, and slm are located.
    :param get_dates: bool, if True return dates.
    :param begin_date: beginning date to load
    :param end_date: ending date to load
    :return: if get_dates:
                cqlm, sqlm, dates_begin, dates_end
            else:
                cqlm, sqlm
    """

    def are_all_num(x: list):
        for i in lmcs_in_queue:
            if x[i - 1].replace('e', '').replace('E', '').replace('E', '').replace('E', '').replace('-',
                                                                                                    '').replace(
                '+', '').replace('.', '').isnumeric():
                pass
            else:
                return False

        return True

    if len(filepath) == 1:
        if filepath[0].is_file():

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
                    if txt_list[i].replace(" ", "").startswith(key):
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

            if get_dates:
                this_date_begin, this_date_end = match_dates_from_filename(filepath[0].name)
                # return clm, slm, [this_date_begin], [this_date_end]
                return SHC(clm, slm), [this_date_begin], [this_date_end]

            else:
                return SHC(clm, slm)

        elif filepath[0].is_dir():
            file_list = FileTool.get_files_in_dir(filepath[0], sub=True)
            files_to_load = []
            for i in range(len(file_list)):
                this_begin_date, this_end_date = match_dates_from_filename(file_list[i].name)
                if this_begin_date >= begin_date and this_end_date <= end_date:
                    files_to_load.append(file_list[i])

            return load_SHC(*files_to_load, key=key, lmax=lmax, lmcs_in_queue=lmcs_in_queue,
                            get_dates=get_dates, begin_date=begin_date, end_date=end_date)

    else:
        shc = None
        dates_begin, dates_end = [], []

        for i in range(len(filepath)):
            load = load_SHC(filepath[i], key=key, lmax=lmax, lmcs_in_queue=lmcs_in_queue,
                            get_dates=get_dates, begin_date=begin_date, end_date=end_date)

            if type(load) is tuple:
                assert len(load) in (1, 3)
                load_shc = load[0]
            else:
                load_shc = load

            if shc is None:
                shc = load_shc
            else:
                shc.append(load_shc)

            if get_dates:
                assert len(load) == 3
                d_begin, d_end = load[1], load[2]
                dates_begin.append(d_begin[0])
                dates_end.append(d_end[0])

        if get_dates:
            return shc, dates_begin, dates_end
        else:
            return shc


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
