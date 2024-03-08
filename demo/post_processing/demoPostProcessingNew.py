import datetime
import enum
import json
import pathlib
import re

from pysrc.auxiliary.preference.EnumClasses import L2InstituteType, L2Release, L2ProductMaxDegree, BasinName, \
    L2LowDegreeType, L2LowDegreeFileID, EmpiricalDecorrelationType, AverageFilterType
from pysrc.auxiliary.aux_tool.FileTool import FileTool


def match_str_enum(enum_class: enum.EnumMeta, value: str):
    assert value in enum_class._member_names_
    match_index = enum_class._member_names_.index(value)
    return list(enum_class)[match_index]


class PostProcessingConfig:
    def __init__(self):
        self.__average_filter_DDK_params = None
        self.__average_filter_non_isotropic_Gaussian_params = None
        self.__average_filter_fan_params = None
        self.__average_filter_Gaussian_params = None
        self.__average_filter_method = None
        self.__de_correlation_window_pnmm_params = None
        self.__de_correlation_window_Duan2009_params_params = None
        self.__de_correlation_window_Wahr2006_params_params = None
        self.__de_correlation_window_stable_params_params = None
        self.__de_correlation_method = None
        self.__low_deg_c30_id = None
        self.__low_deg_c30_replace = False
        self.__low_deg_c20_id = None
        self.__low_deg_c20_replace = False
        self.__low_deg_d1_id = None
        self.__low_deg_d1_replace = False
        self.__date_ending = None
        self.__date_beginning = None
        self.__basin_preset = None
        self.__basin_mode = None
        self.__basin_local_path = None
        self.__data_source_local_path = None
        self.__data_source_lmax = None
        self.__data_source_release = None
        self.__data_source_institute = None
        self.__data_source_mode = None

    def set_from_json(self, filepath: pathlib.WindowsPath):
        """
        :param filepath: json filepath, or a dict loaded from json.
        """
        assert type(filepath) in (pathlib.WindowsPath,)

        with open(filepath, 'r') as f:
            dict_from_json = json.load(f)

        '''set data source'''
        assert dict_from_json['data source']['mode'] in ('presets', 'local path')
        self.__data_source_mode = dict_from_json['data source']['mode']

        if self.__data_source_mode == "presets":
            self.set_data_source(
                institute=match_str_enum(
                    L2InstituteType,
                    dict_from_json['data source']['institute']
                )
            )

            self.set_data_source(
                release=match_str_enum(
                    L2InstituteType,
                    dict_from_json['data source']['release']
                )
            )

            self.set_data_source(
                lmax=match_str_enum(
                    L2ProductMaxDegree,
                    'Degree' + str(dict_from_json['Data source']['lmax'])
                )
            )

        elif self.__data_source_mode == "local path":
            self.set_data_source(
                local_path=FileTool.get_project_dir(dict_from_json['data source']['local path'])
            )

        else:
            return -1

        '''set dates'''

        def get_date_from_str(s):
            re_match = re.match(r'(\d{4})-(\d{1,2})-*(\d{1,2})*', s)
            year, month, day = re_match.groups()
            if day is None:
                day = 1
            return datetime.date(int(year), int(month), int(day))

        self.set_date(
            beginning=get_date_from_str(dict_from_json['dates']['beginning']),
            ending=get_date_from_str(dict_from_json['dates']['ending'])
        )

        '''set basin'''
        assert dict_from_json['basin']['mode'] in ('presets', 'local path')
        self.__basin_mode = dict_from_json['basin']['mode']

        if self.__basin_mode == "presets":
            self.set_basin(
                preset_basin=match_str_enum(
                    BasinName,
                    dict_from_json['basin']['preset basin']
                )
            )

        elif self.__basin_mode == "local path":
            self.set_basin(
                local_path=
                FileTool.get_project_dir(dict_from_json['basin']['local path'])
            )

        else:
            return -1

        '''set replacing low degrees'''
        self.set_replace_low_deg(
            degree1=dict_from_json['replace low degree']['replace degree 1'],
            c20=dict_from_json['replace low degree']['replace c20'],
            c30=dict_from_json['replace low degree']['replace c30'],
            degree1_id=match_str_enum(L2LowDegreeFileID, dict_from_json['replace low degree']['degree1 file id']),
            c20_id=match_str_enum(L2LowDegreeFileID, dict_from_json['replace low degree']['c20 file id']),
            c30_id=match_str_enum(L2LowDegreeFileID, dict_from_json['replace low degree']['c30 file id'])
        )

        '''set de-correlation'''
        self.set_de_correlation(
            method=None if dict_from_json['de-correlation']['method'] == 'none' else match_str_enum(
                EmpiricalDecorrelationType, dict_from_json['de-correlation']['method']
            ),
            pnmm_params=dict_from_json['de-correlation']['PnMm params'],
            window_stable_params=dict_from_json['de-correlation']['window stable params'],
            window_Wahr2006_params=dict_from_json['de-correlation']['window Wahr2006 params'],
            window_Duan2009_params=dict_from_json['de-correlation']['window Duan2009 params'],
        )

        '''set average filter'''
        self.set_average_filter(
            method=None if dict_from_json['average filter']['method'] == 'none' else match_str_enum(
                AverageFilterType, dict_from_json['average filter']['method']
            ),
            Gaussian_params=dict_from_json['average filter']['Gaussian params'],
            fan_params=dict_from_json['average filter']['fan params'],
            non_isotropic_Gaussian_params=dict_from_json['average filter']['non_isotropic_Gaussian params'],
            DDK_params=dict_from_json['average filter']['DDK params'],
        )

        return self

    def set_data_source(self, institute=None, release=None, lmax=None, local_path=None):
        assert ((institute is not None) or (release is not None) or (lmax is not None)) ^ (local_path is not None)

        if local_path is not None:
            self.__data_source_mode = 'local path'

            self.__data_source_local_path = local_path

        elif institute is not None or release is not None or lmax is not None:
            self.__data_source_mode = 'presets'

            if institute is not None:
                self.__data_source_institute = institute

            if release is not None:
                self.__data_source_release = release

            if lmax is not None:
                self.__data_source_lmax = lmax

        else:
            return -1

        return self

    def get_data_source(self):
        return dict(
            mode=self.__data_source_mode,
            institute=self.__data_source_institute,
            release=self.__data_source_release,
            lmax=self.__data_source_lmax,
            local_path=self.__data_source_local_path
        )

    def set_date(self, beginning=None, ending=None):
        if beginning is not None:
            self.__date_beginning = beginning

        if ending is not None:
            self.__date_ending = ending

    def get_dates(self):
        return dict(
            begin=self.__date_beginning,
            end=self.__date_ending
        )

    def set_basin(self, preset_basin=None, local_path=None):
        assert (preset_basin is not None) ^ (local_path is not None)

        if local_path is not None:
            self.__basin_mode = 'local path'

            self.__basin_local_path = local_path

        elif preset_basin is not None:
            self.__basin_mode = 'presets'

            self.__basin_preset = preset_basin

        return self

    def get_basin(self):
        return dict(
            mode=self.__basin_mode,
            preset_basin=self.__basin_preset,
            local_path=self.__basin_local_path
        )

    def set_replace_low_deg(
            self, degree1=None, degree1_id=L2LowDegreeFileID.TN13,
            c20=None, c20_id=L2LowDegreeFileID.TN14,
            c30=None, c30_id=L2LowDegreeFileID.TN14
    ):

        if degree1 is not None:
            self.__low_deg_d1_replace = degree1

        if c20 is not None:
            self.__low_deg_c20_replace = c20

        if c30 is not None:
            self.__low_deg_c30_replace = c30

        self.__low_deg_d1_id = degree1_id
        self.__low_deg_c20_id = c20_id
        self.__low_deg_c30_id = c30_id
        return self

    def get_replace_low_deg(self):
        return dict(
            degree1=self.__low_deg_d1_replace,
            degree1_id=self.__low_deg_d1_id,
            c20=self.__low_deg_c20_replace,
            c20_id=self.__low_deg_c20_id,
            c30=self.__low_deg_c30_replace,
            c30_id=self.__low_deg_c30_id,
        )

    def set_de_correlation(
            self, method=None,
            pnmm_params=(3, 5),
            window_stable_params=(3, 5, 5),
            window_Wahr2006_params=(3, 5, 5, 10, 30),
            window_Duan2009_params=(3, 5, 5, 10, 30, 10, 30),
    ):
        self.__de_correlation_method = method
        self.__de_correlation_window_pnmm_params = pnmm_params
        self.__de_correlation_window_stable_params_params = window_stable_params
        self.__de_correlation_window_Wahr2006_params_params = window_Wahr2006_params
        self.__de_correlation_window_Duan2009_params_params = window_Duan2009_params

        return self

    def get_de_correlation(self):
        return dict(
            method=self.__de_correlation_method,
            pnmm_params=self.__de_correlation_window_pnmm_params,
            window_stable_params=self.__de_correlation_window_stable_params_params,
            window_Wahr2006_params=self.__de_correlation_window_Wahr2006_params_params,
            window_Duan2009_params=self.__de_correlation_window_Duan2009_params_params,
        )

    def set_average_filter(
            self, method=None,
            Gaussian_params=300,
            fan_params=(300, 300),
            non_isotropic_Gaussian_params=(300, 500, 20),
            DDK_params=3,
    ):
        self.__average_filter_method = method
        self.__average_filter_Gaussian_params = Gaussian_params
        self.__average_filter_fan_params = fan_params
        self.__average_filter_non_isotropic_Gaussian_params = non_isotropic_Gaussian_params
        self.__average_filter_DDK_params = DDK_params

        return self

    def get_average_filter(self):
        return dict(
            method=self.__average_filter_method,
            Gaussian_params=self.__average_filter_Gaussian_params,
            fan_params=self.__average_filter_fan_params,
            non_isotropic_Gaussian_params=self.__average_filter_non_isotropic_Gaussian_params,
            DDK_params=self.__average_filter_DDK_params,
        )


if __name__ == '__main__':
    ppconfig = PostProcessingConfig().set_from_json(
        FileTool.get_project_dir("setting/post_processing/PostProcessingNew.json"))

    print(ppconfig.get_data_source())
    print(ppconfig.get_basin())
    print(ppconfig.get_dates())
    print(ppconfig.get_replace_low_deg())
    print(ppconfig.get_de_correlation())
    print(ppconfig.get_average_filter())
