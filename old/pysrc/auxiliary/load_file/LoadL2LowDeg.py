import datetime
import pathlib
import re

import numpy as np

import pysrc.auxiliary.preference.EnumClasses as Enum
from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool


class LoadLowDegreeConfig:
    def __init__(self):
        self.file_id = Enum.L2LowDegreeFileID.TN11
        self.institute = Enum.L2InstituteType.CSR
        self.release = Enum.L2Release.RL061

    def set_file_id(self, file_id: Enum.L2LowDegreeFileID):
        self.file_id = file_id
        return self

    def set_institute(self, institute: Enum.L2InstituteType):
        self.institute = institute
        return self

    def set_release(self, release: Enum.L2Release):
        self.release = release
        return self

    def get_path(self):
        return FileTool.get_l2_low_deg_path(
            self.file_id,
            self.institute,
            self.release
        )


class LoadLowDegree:
    """
    This class is to load auxiliary low-degree GSM products.

    self.results is supposed to be a dict, including:
        'c10': list in shape (2, length). dimension 1: datetime.date; dimension 2: values.
        'c11':
        's11':
        'c20':
        'c30':
    also, each param can be called respectively, like self.c10   """

    def __init__(self):
        self.configuration = LoadLowDegreeConfig()

        self.results = {}
        self.c10, self.c11, self.s11 = None, None, None
        self.c20 = None
        self.c30 = None

        self.c10_dev, self.c11_dev, self.s11_dev = None, None, None
        self.c20_dev = None
        self.c30_dev = None

        pass

    def config(self, config: LoadLowDegreeConfig):
        self.configuration = config

        return self

    def _load(self):
        filepath = self.configuration.get_path()

        if 'TN-11' in filepath.name:
            # with open(filepath) as f:
            #     txt = f.read()
            #
            # dates_c20 = []
            # values_c20 = []
            # values_c20_dev = []
            #
            # pat_data = r'\s*^\d{5}.*'
            # data = re.findall(pat_data, txt, re.M)
            # for i in data:
            #     line = i.split()
            #     ave_dates = TimeTool.convert_date_format(
            #         (float(line[0]) + float(line[5])) / 2,
            #         input_type=TimeTool.DateFormat.MJD,
            #         output_type=TimeTool.DateFormat.ClassDate
            #     )
            #
            #     dates_c20.append(ave_dates)
            #     values_c20.append(float(line[2]))
            #     values_c20_dev.append(float(line[4]) * 1e-10)
            #
            # c20 = [dates_c20, np.array(values_c20)]
            # c20_dev = [dates_c20, np.array(values_c20_dev)]
            #
            # self.results.update(dict(c20=c20, c20_dev=c20_dev))
            # self.c20 = c20
            # self.c20_dev = c20_dev
            this_load = load_TN11(filepath)
            self.results.update(this_load)
            self.c20 = this_load['c20']
            self.c20_dev = this_load['c20_dev']

            return self

        elif 'TN-13' in filepath.name:
            # with open(filepath) as f:
            #     txt = f.read()
            #
            # times_c10 = []
            # values_c10 = []
            # values_c10_dev = []
            #
            # times_c11 = []
            # values_c11 = []
            # values_c11_dev = []
            #
            # times_s11 = []
            # values_s11 = []
            # values_s11_dev = []
            #
            # pat_data = r'^GRCOF2.*\w'
            # data = re.findall(pat_data, txt, re.M)
            # for i in data:
            #     line = i.split()
            #     m = int(line[2])
            #     ymd_begin = line[7][:8]
            #     date_begin = datetime.date(int(ymd_begin[:4]), int(ymd_begin[4:6]), int(ymd_begin[6:]))
            #     mjd_begin = TimeTool.convert_date_format(
            #         date_begin,
            #         input_type=TimeTool.DateFormat.ClassDate,
            #         output_type=TimeTool.DateFormat.MJD,
            #     )
            #
            #     ymd_end = line[8][:8]
            #     date_end = datetime.date(int(ymd_end[:4]), int(ymd_end[4:6]), int(ymd_end[6:]))
            #     mjd_end = TimeTool.convert_date_format(
            #         date_end,
            #         input_type=TimeTool.DateFormat.ClassDate,
            #         output_type=TimeTool.DateFormat.MJD,
            #     )
            #
            #     ave_dates = TimeTool.convert_date_format(
            #         (mjd_begin + mjd_end) / 2,
            #         input_type=TimeTool.DateFormat.MJD,
            #         output_type=TimeTool.DateFormat.ClassDate,
            #     )
            #
            #     if m == 0:
            #         times_c10.append(ave_dates)
            #         values_c10.append(float(line[3]))
            #         values_c10_dev.append(float(line[5]))
            #
            #     elif m == 1:
            #         times_c11.append(ave_dates)
            #         values_c11.append(float(line[3]))
            #         values_c11_dev.append(float(line[5]))
            #
            #         times_s11.append(ave_dates)
            #         values_s11.append(float(line[4]))
            #         values_s11_dev.append(float(line[6]))
            #
            # c10 = [times_c10, np.array(values_c10)]
            # c11 = [times_c11, np.array(values_c11)]
            # s11 = [times_s11, np.array(values_s11)]
            #
            # c10_dev = [times_c10, np.array(values_c10_dev)]
            # c11_dev = [times_c11, np.array(values_c11_dev)]
            # s11_dev = [times_s11, np.array(values_s11_dev)]

            this_load = load_TN13(filepath)
            self.results.update(this_load)
            self.c10 = this_load['c10']
            self.c11 = this_load['c11']
            self.s11 = this_load['s11']

            self.c10_dev = this_load['c10_dev']
            self.c11_dev = this_load['c11_dev']
            self.s11_dev = this_load['s11_dev']

            return self

        elif 'TN-14' in filepath.name:
            # with open(filepath) as f:
            #     txt = f.read()
            #
            # dates_c20 = []
            # values_c20 = []
            # values_c20_dev = []
            #
            # dates_c30 = []
            # values_c30 = []
            # values_c30_dev = []
            #
            # pat_data = r'\s*^\d{5}.*'
            # data = re.findall(pat_data, txt, re.M)
            # for i in data:
            #     line = i.split()
            #     ave_dates = TimeTool.convert_date_format(
            #         (float(line[0]) + float(line[8])) / 2,
            #         input_type=TimeTool.DateFormat.MJD,
            #         output_type=TimeTool.DateFormat.ClassDate,
            #     )
            #
            #     if line[2] != 'NaN':
            #         dates_c20.append(ave_dates)
            #         values_c20.append(float(line[2]))
            #         values_c20_dev.append(float(line[4]) * 1e-10)
            #
            #     if line[5] != 'NaN':
            #         dates_c30.append(ave_dates)
            #         values_c30.append(float(line[5]))
            #         values_c30_dev.append(float(line[7]) * 1e-10)
            #
            # c20 = [dates_c20, np.array(values_c20)]
            # c30 = [dates_c30, np.array(values_c30)]
            # c20_dev = [dates_c20, np.array(values_c20_dev)]
            # c30_dev = [dates_c30, np.array(values_c30_dev)]

            this_load = load_TN14(filepath)
            self.results.update(this_load)
            self.c20 = this_load['c20']
            self.c30 = this_load['c30']
            self.c20_dev = this_load['c20_dev']
            self.c30_dev = this_load['c30_dev']

            return self

        else:
            print('check low-degree file path.')
            return self

    def get_degree1(self):
        self._load()
        assert self.c10 is not None
        assert self.c11 is not None
        assert self.s11 is not None

        return dict(
            c10=self.c10,
            c11=self.c11,
            s11=self.s11,
        )

    def get_c20(self):
        self._load()
        assert self.c20 is not None

        return dict(
            c20=self.c20
        )

    def get_c30(self):
        self._load()
        assert self.c30 is not None

        return dict(
            c30=self.c30
        )

    def get_results(self):
        """
        return: dict, {'c10': array [ [yyyymmdd, ...], [value, ...] ], ...}
        """
        self._load()

        result_dict = {}

        low_deg_result = (
            self.c10, self.c11, self.s11, self.c20, self.c30,
            self.c10_dev, self.c11_dev, self.s11_dev, self.c20_dev, self.c30_dev
        )
        low_deg_name = (
            'c10', 'c11', 's11', 'c20', 'c30',
            'c10_dev', 'c11_dev', 's11_dev', 'c20_dev', 'c30_dev'
        )
        for i in range(len(low_deg_result)):
            if low_deg_result[i] is not None:
                result_dict[low_deg_name[i]] = low_deg_result[i]

        return result_dict


def load_TN11(filepath):
    with open(filepath) as f:
        txt = f.read()

    dates_c20 = []
    values_c20 = []
    values_c20_dev = []

    pat_data = r'\s*^\d{5}.*'
    data = re.findall(pat_data, txt, re.M)
    for i in data:
        line = i.split()

        ave_dates = TimeTool.convert_date_format(
            (float(line[0]) + float(line[5])) / 2,
            input_type=TimeTool.DateFormat.MJD,
            output_type=TimeTool.DateFormat.ClassDate
        )

        dates_c20.append(ave_dates)
        values_c20.append(float(line[2]))
        values_c20_dev.append(float(line[4]) * 1e-10)

    c20 = [dates_c20, np.array(values_c20)]
    c20_dev = [dates_c20, np.array(values_c20_dev)]

    result = dict(
        c20=c20,
        c20_dev=c20_dev
    )
    return result


def load_TN13(filepath):
    with open(filepath) as f:
        txt = f.read()

    times_c10 = []
    values_c10 = []
    values_c10_dev = []

    times_c11 = []
    values_c11 = []
    values_c11_dev = []

    times_s11 = []
    values_s11 = []
    values_s11_dev = []

    pat_data = r'^GRCOF2.*\w'
    data = re.findall(pat_data, txt, re.M)
    for i in data:
        line = i.split()
        m = int(line[2])
        ymd_begin = line[7][:8]
        date_begin = datetime.date(int(ymd_begin[:4]), int(ymd_begin[4:6]), int(ymd_begin[6:]))
        mjd_begin = TimeTool.convert_date_format(
            date_begin,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.MJD,
        )

        ymd_end = line[8][:8]
        date_end = datetime.date(int(ymd_end[:4]), int(ymd_end[4:6]), int(ymd_end[6:]))
        mjd_end = TimeTool.convert_date_format(
            date_end,
            input_type=TimeTool.DateFormat.ClassDate,
            output_type=TimeTool.DateFormat.MJD,
        )

        ave_dates = TimeTool.convert_date_format(
            (mjd_begin + mjd_end) / 2,
            input_type=TimeTool.DateFormat.MJD,
            output_type=TimeTool.DateFormat.ClassDate,
        )

        if m == 0:
            times_c10.append(ave_dates)
            values_c10.append(float(line[3]))
            values_c10_dev.append(float(line[5]))

        elif m == 1:
            times_c11.append(ave_dates)
            values_c11.append(float(line[3]))
            values_c11_dev.append(float(line[5]))

            times_s11.append(ave_dates)
            values_s11.append(float(line[4]))
            values_s11_dev.append(float(line[6]))

    c10 = [times_c10, np.array(values_c10)]
    c11 = [times_c11, np.array(values_c11)]
    s11 = [times_s11, np.array(values_s11)]

    c10_dev = [times_c10, np.array(values_c10_dev)]
    c11_dev = [times_c11, np.array(values_c11_dev)]
    s11_dev = [times_s11, np.array(values_s11_dev)]

    result = dict(
        c10=c10, c11=c11, s11=s11,
        c10_dev=c10_dev, c11_dev=c11_dev, s11_dev=s11_dev
    )
    return result


def load_TN14(filepath):
    with open(filepath) as f:
        txt = f.read()

    dates_c20 = []
    values_c20 = []
    values_c20_dev = []

    dates_c30 = []
    values_c30 = []
    values_c30_dev = []

    pat_data = r'\s*^\d{5}.*'
    data = re.findall(pat_data, txt, re.M)
    for i in data:
        line = i.split()
        ave_dates = TimeTool.convert_date_format(
            (float(line[0]) + float(line[8])) / 2,
            input_type=TimeTool.DateFormat.MJD,
            output_type=TimeTool.DateFormat.ClassDate,
        )

        if line[2] != 'NaN':
            dates_c20.append(ave_dates)
            values_c20.append(float(line[2]))
            values_c20_dev.append(float(line[4]) * 1e-10)

        if line[5] != 'NaN':
            dates_c30.append(ave_dates)
            values_c30.append(float(line[5]))
            values_c30_dev.append(float(line[7]) * 1e-10)

    c20 = [dates_c20, np.array(values_c20)]
    c30 = [dates_c30, np.array(values_c30)]
    c20_dev = [dates_c20, np.array(values_c20_dev)]
    c30_dev = [dates_c30, np.array(values_c30_dev)]

    result = dict(
        c20=c20, c30=c30,
        c20_dev=c20_dev, c30_dev=c30_dev
    )
    return result


def load_low_degs(
        *filepath: pathlib.Path,
        file_id: Enum.L2LowDegreeFileID = None,
        release: Enum.L2Release = None,
        institute: Enum.L2InstituteType = None,
):
    if len(filepath) == 0:
        assert file_id in Enum.L2LowDegreeFileID
        this_filepath = FileTool.get_l2_low_deg_path(
            file_id=file_id, release=release, institute=institute
        )
        return load_low_degs(this_filepath)

    if len(filepath) == 1:
        this_filepath = filepath[0]

        if this_filepath.is_file():
            check_ids = ("TN-11", "TN-13", "TN-14")
            check_pattern = "(" + ")|(".join(check_ids) + ")"  # r"(TN-11)|(TN-13)|(TN-14)"
            checked = re.search(check_pattern, this_filepath.name) is not None
            if not checked:
                assert False, f"file name should include one of ids: {check_pattern}"

            if "TN-11" in this_filepath.name:
                return load_TN11(this_filepath)
            elif "TN-13" in this_filepath.name:
                return load_TN13(this_filepath)
            elif "TN-14" in this_filepath.name:
                return load_TN14(this_filepath)
            else:
                assert False

        elif this_filepath.is_dir():
            assert file_id in Enum.L2LowDegreeFileID

            filepath_to_load = FileTool.get_l2_low_deg_path(
                this_filepath, file_id=file_id, release=release, institute=institute
            )

            return load_low_degs(filepath_to_load)

    else:
        result = {}
        for i in range(len(filepath)):
            this_result = load_low_degs(filepath[i])
            result.update(this_result)

        return result

    pass


def demo():
    low_degs = load_low_degs(file_id=Enum.L2LowDegreeFileID.TN13, institute=Enum.L2InstituteType.CSR,
                             release=Enum.L2Release.RL06)  # load degree-1
    low_degs.update(load_low_degs(file_id=Enum.L2LowDegreeFileID.TN14))  # load c20 and c30


if __name__ == '__main__':
    demo()
