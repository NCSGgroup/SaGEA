import enum
import json
import pathlib

from pysrc.auxiliary.preference.EnumClasses import L2InstituteType, L2Release, L2ProductMaxDegree
from pysrc.auxiliary.tools.FileTool import FileTool


def match_str_enum(enum_class: enum.EnumMeta, value: str):
    match_index = enum_class._member_names_.index(value)
    return list(enum_class)[match_index]


class PostProcessingConfig:
    def __init__(self):
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
        assert dict_from_json['Data source']['mode'] in ('presets', 'local path')
        self.__data_source_mode = dict_from_json['Data source']['mode']

        if self.__data_source_mode == "presets":
            assert dict_from_json['Data source']['institute'] in ('CSR', 'JPL', 'GFZ')
            self.__data_source_institute = match_str_enum(
                L2InstituteType,
                dict_from_json['Data source']['institute']
            )

            assert dict_from_json['Data source']['release'] in ('RL06', 'RL061', 'RL062')
            self.__data_source_release = match_str_enum(
                L2Release,
                dict_from_json['Data source']['release']
            )

            assert dict_from_json['Data source']['lmax'] in (60, 90, 96)
            self.__data_source_lmax = match_str_enum(
                L2ProductMaxDegree,
                'Degree' + str(dict_from_json['Data source']['lmax'])
            )

        elif self.__data_source_mode == "local path":
            self.__data_source_local_path = FileTool.get_project_dir(dict_from_json['Data source']['local path'])

        else:
            return -1

        return self

    def get_data_source(self):
        return (
            self.__data_source_mode,
            self.__data_source_institute,
            self.__data_source_release,
            self.__data_source_lmax,
            self.__data_source_local_path

        )


if __name__ == '__main__':
    ppconfig = PostProcessingConfig().set_from_json(
        FileTool.get_project_dir("setting/post_processing/PostProcessingNew.json"))

    print(ppconfig.get_data_source())
