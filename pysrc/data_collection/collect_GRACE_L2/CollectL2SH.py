import ftplib
import re
import time
import datetime
import json
from pathlib import Path, WindowsPath

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool

from pysrc.auxiliary.preference.EnumClasses import Satellite, L2DataServer, L2ProductType, L2InstituteType, L2Release, \
    L2ProductMaxDegree
from pysrc.auxiliary.scripts.MatchConfigWithEnums import match_config


class CollectL2SHConfig:
    def __init__(self):
        """default setting"""
        self.server = L2DataServer.GFZ
        self.satellite = Satellite.GRACE
        self.product_type = L2ProductType.GSM
        self.institute = L2InstituteType.CSR
        self.release = L2Release.RL06
        self.degree = L2ProductMaxDegree.Degree60
        self.beginning_date = datetime.date(2002, 4, 1)
        self.ending_date = datetime.date(2002, 12, 31)
        self.update_mode = False

        self.__max_relink_time = 3

    def set_server(self, server: L2DataServer):
        self.server = server
        return self

    def set_product_type(self, product_type: L2ProductType):
        self.product_type = product_type
        return self

    def set_institute(self, institute: L2InstituteType):
        self.institute = institute
        return self

    def set_release(self, release: L2Release):
        self.release = release
        return self

    def set_satellite(self, sat: Satellite):
        self.satellite = sat
        return self

    def set_degree(self, deg: L2ProductMaxDegree):
        self.degree = deg
        return self

    def set_beginning_date(self, beginning_date: datetime.date or str):
        """
        :param beginning_date: datetime.date, or str like '2002-04'
        """
        self._set_date(beginning_date, 'beginning')
        return self

    def set_ending_date(self, ending_date: datetime.date or str):
        """
        :param ending_date: datetime.date, or str like '2015-12'
        """
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
            yearmonth = date.split('-')
            yearmonth = tuple((int(yearmonth[i]) for i in range(len(yearmonth))))  # year, month
            assert len(yearmonth) == 2

            if beginning_or_ending == 'beginning':
                self.beginning_date = datetime.date(*yearmonth, 1)
            else:
                self.ending_date = TimeTool.get_the_final_day_of_this_month(datetime.date(*yearmonth, 1))

    def set_update_mode(self, mode: bool):
        self.update_mode = mode

    def set_from_json(self, path: Path):
        """
        This function is to set or update parameters through external JSON files. The JSON file must contain these
        variables: "server", "satellite", "product_type", "institute" and "release", "beginning_date", "ending_date"
        and "update_mode".
        :param path: a standardized class of pathlib.Path, and the filename should end with ".json"
        :return:
        """

        # ensure that the path describes a JSON file
        assert type(path) is WindowsPath and path.name.endswith('.json')

        # load json file to a dict
        with open(path) as jsonfile:
            json_dict = json.load(jsonfile)

        # ensure that the necessary parameters are included in the JSON file
        assert {'server', 'satellite', 'product_type', 'institute', 'release', 'beginning_date', 'ending_date',
                'update_mode'} <= set(json_dict.keys())

        # transmitting parameters for enum classes
        enum_classes = (L2DataServer, Satellite, L2ProductType, L2InstituteType, L2Release)
        json_keys = ('server', 'satellite', 'product_type', 'institute', 'release')
        setting_functions = (
            self.set_server, self.set_satellite, self.set_product_type, self.set_institute, self.set_release)

        match_config(json_dict, json_keys, enum_classes, setting_functions)

        # transmitting parameters for not enum classes
        self.set_beginning_date(json_dict['beginning_date'])
        self.set_ending_date(json_dict['ending_date'])
        self.set_update_mode((True, False)[
                                 ('on', 'off').index(json_dict['update_mode'])
                             ])

        return self

    def set_max_relink_times(self, times):
        self.__max_relink_time = times
        return self

    def get_max_relink_times(self):
        return self.__max_relink_time

    def __str__(self):
        """

        :return: str, information of the configuration
        """

        info = f'Data Server:\t{self.server.name}\n' \
               f'Satellite:\t{self.satellite.name}\n' \
               f'L2 Product Type:\t{self.product_type.name}\n' \
               f'Solution Institute:\t{self.institute.name}\n' \
               f'L2 Data Release:\t{self.release.name}\n' \
               f'Beginning date:\t{self.beginning_date}\n' \
               f'Ending date:\t{self.ending_date}\n' \
               f'Update mode:\t{self.update_mode}'

        return info


class CollectL2SH:
    def __init__(self, preset: CollectL2SHConfig = None):
        """
        presets can be input later in function .config(self, presets)
        :param preset: a standardized class of CollectL2ProductConfig or pathlib.Path;
                        a default presets of class CollectL2ProductConfig would be used if presets got None.
        """
        # in form of an instantiated class CollectL2ProductConfig
        self.configuration = CollectL2SHConfig()

        self._ftp = None

        self.__max_relink_time = self.configuration.get_max_relink_times()
        self.__relink_time = 0
        self.__sleep_seconds_before_relink = 5

        if preset is not None:
            self.config(preset)

    def config(self, preset: CollectL2SHConfig or Path):
        """
        This function is to set parameters for collecting GRACE level-2 data, including satellite data server, product
        type, publishing institute, and publishing release. These parameters can be set uniformly by inputting a
        standardized instantiated class or a path of JSON file.
        :param preset: a standardized class of CollectL2ProductConfig or pathlib.Path
        :return:
        """

        # load the presets
        assert type(preset) in (WindowsPath, CollectL2SHConfig)
        if type(preset) is WindowsPath:
            configuration = CollectL2SHConfig().set_from_json(preset)
        elif type(preset) is CollectL2SHConfig:
            configuration = preset
        else:
            raise Exception

        self.configuration = configuration

        return self

    def _get_hostname(self):
        if self.configuration.server == L2DataServer.GFZ:
            hostname = 'isdcftp.gfz-potsdam.de'

        elif self.configuration.server == L2DataServer.ITSG:
            hostname = 'ftp.tugraz.at'

        else:
            raise Exception

        return hostname

    def _login(self):
        hostname = self._get_hostname()

        self._ftp = ftplib.FTP(hostname)
        self._ftp.login()

        return self

    def _get_remote_l2dir(self):
        if self.configuration.server == L2DataServer.GFZ:
            sat_str = self.configuration.satellite.name.lower().replace('_', '-')

            if self.configuration.release.name in ('RL061', 'RL062'):
                path = Path(
                    sat_str) / 'Level-2' / self.configuration.institute.name / self.configuration.release.name.replace(
                    'RL06', 'RL06.')
            else:
                path = Path(
                    sat_str) / 'Level-2' / self.configuration.institute.name / self.configuration.release.name

        elif self.configuration.server == L2DataServer.ITSG:
            path = Path('outgoing/ITSG/GRACE')
            path /= self.configuration.release.name.replace('ITSGGrace', 'ITSG-Grace')
            path /= 'monthly'
            path /= f'monthly_n{str(self.configuration.degree.value)}'

        else:
            raise Exception

        return path

    def _get_local_l2path_from_filename(self, filename):
        """
        get local path 'projectname/data/x' based on a (remote) filename
        :return: pathlib.Path, local filepath to save the file
        """
        local_l2dir = FileTool.get_project_dir() / 'data/L2_SH_products'
        formatted_filename = self._format_filename(filename=filename)

        local_l2dir /= formatted_filename['product_type']
        local_l2dir /= formatted_filename['solution_institute']
        local_l2dir /= formatted_filename['product_release']
        local_l2dir /= formatted_filename['product_id']
        local_l2dir /= str(formatted_filename['beginning_date'].year)

        return local_l2dir / filename.replace('.gz', '')

    def _format_filename(self, filename):
        if self.configuration.server == L2DataServer.GFZ:
            regular_pattern = r'([A-Z]{3})-2_(\d{7})-(\d{7})_([A-Z]{4})_([A-Z]{5})_(.{4})_(\d{4}).gz'
            matches = re.match(regular_pattern, filename).groups()
            # parameter matches is a tuple, the elements in order are:
            # index 0, product type, 'GSM' or 'GAX'
            # index 1, beginning day (year and days of the year), e.g. '2002095'
            # index 2, ending day (year and days of the year), e.g. '2002120'
            # index 3, satellite short name, 'GRAC' for GRACE, 'GRFO' for GRACE_FO
            # index 4, solution institute name, 'UTCSR' for CSR, 'GFZOP' for GFZ, 'JPLEM' for JPL
            # index 5, product id, e.g. 'BA01'
            # index 6, product release, e.g. '0600'
            return dict(
                product_type=matches[0],
                beginning_date=TimeTool.convert_date_format(matches[1], input_type=TimeTool.DateFormat.YearDay,
                                                            output_type=TimeTool.DateFormat.ClassDate),
                ending_date=TimeTool.convert_date_format(matches[2], input_type=TimeTool.DateFormat.YearDay,
                                                         output_type=TimeTool.DateFormat.ClassDate),
                satellite_short_name=(Satellite.GRACE.name, Satellite.GRACE_FO.name)[
                    ('GRAC', 'GRFO').index(matches[3])
                ],
                solution_institute=(L2InstituteType.CSR.name, L2InstituteType.GFZ.name, L2InstituteType.JPL.name)[
                    ('UTCSR', 'GFZOP', 'JPLEM').index(matches[4])
                ],
                product_id=matches[5],
                product_release=(L2Release.RL06.name, L2Release.RL061.name, L2Release.RL062.name,)[
                    ('0600', '0601', '0602',).index(matches[6])
                ]
            )

        elif self.configuration.server == L2DataServer.ITSG:
            regular_pattern = r'ITSG-(.*)_(n\d{2})_(\d{4})-(\d{2}).gfc'
            matches = re.match(regular_pattern, filename).groups()
            # parameter matches is a tuple, the elements in order are:
            # index 0, product release, 'Grace_operational'
            # index 1, max degree/order, 'n60'
            # index 2, year, e.g. '2018'
            # index 3, month, e.g. '06'
            ave_year, ave_month = int(matches[2]), int(matches[3])
            beginning_date = datetime.date(ave_year, ave_month, 1)
            ending_date = TimeTool.get_the_final_day_of_this_month(beginning_date)

            return dict(
                product_type='GSM',
                beginning_date=beginning_date,
                ending_date=ending_date,
                satellite_short_name=Satellite.GRACE_FO.name if self.configuration.release == L2Release.ITSGGrace_operational else Satellite.GRACE.name,
                solution_institute=L2InstituteType.ITSG.name,
                product_id=matches[1],
                product_release=matches[0]

            )

        else:
            raise Exception

    def _into_remote_filepath(self):
        path = self._get_remote_l2dir()
        path_str = str(path).replace('\\', '/')

        self._ftp.cwd(path_str)

        return self

    def run(self):
        initial_update_mode = self.configuration.update_mode
        # This method may temporarily change the settings set by the user,
        # therefore here make a backup before running for subsequent restoration of the settings.

        # make a dir "temp"
        dir_temp = FileTool.get_project_dir('temp')
        if not dir_temp.exists():
            dir_temp.mkdir(parents=True)

        if self.configuration.server == L2DataServer.GFZ:
            for relink_time in range(self.__max_relink_time):
                try:
                    self._run_once_for_server_gfz()

                except Exception as e:
                    self.__relink_time += 1
                    if self.__relink_time >= self.__max_relink_time:
                        raise Exception(f'The relinking time has reached to the maximum ({self.__max_relink_time}), '
                                        f'please check.')

                    print(f'There were some issues: "{e}", relinking for the {self.__relink_time}th time...')
                    self._ftp.quit()

                    time.sleep(self.__sleep_seconds_before_relink)
                    self.configuration.set_update_mode(True)

                else:
                    break

        elif self.configuration.server == L2DataServer.ITSG:
            for relink_time in range(self.__max_relink_time):
                try:
                    self._run_once_for_server_itsg()

                except Exception as e:
                    self.__relink_time += 1
                    if self.__relink_time >= self.__max_relink_time:
                        raise Exception(f'The relinking time has reached to the maximum ({self.__max_relink_time}), '
                                        f'please check.')

                    print(f'There were some issues: "{e}", relinking for the {self.__relink_time}th time...')
                    self._ftp.quit()

                    time.sleep(self.__sleep_seconds_before_relink)
                    self.configuration.set_update_mode(True)

                else:
                    break

        else:
            raise Exception

        self.configuration.set_update_mode(initial_update_mode)
        return self

    def _run_once_for_server_gfz(self):
        print('Connecting the remote server...')
        self._login()
        self._into_remote_filepath()

        files_list = self._ftp.nlst()
        files_list.sort()

        for i in range(len(files_list)):
            this_file = files_list[i]

            formatted_filename = self._format_filename(filename=this_file)
            if formatted_filename['product_type'] != self.configuration.product_type.name:
                continue
            if formatted_filename['beginning_date'] < self.configuration.beginning_date:
                continue
            if formatted_filename['ending_date'] > self.configuration.ending_date:
                continue
            if formatted_filename['product_release'] != self.configuration.release.name:
                continue

            temp_gz_filepath = FileTool.get_project_dir() / 'temp'
            temp_gz_filepath /= this_file

            local_l2path = self._get_local_l2path_from_filename(this_file)
            if self.configuration.update_mode and Path.exists(local_l2path):
                print(f'File {local_l2path.name} already exists at {local_l2path.parent}.')
                continue

            if not Path.exists(local_l2path.parent):
                Path.mkdir(local_l2path.parent, parents=True)

            with open(temp_gz_filepath, "wb") as f:
                file_handle = f.write

                print(f'Collecting {this_file}...', end=' ')
                self._ftp.retrbinary('RETR %s' % this_file, file_handle, blocksize=1024)

            FileTool.un_gz(temp_gz_filepath, local_l2path)
            Path.unlink(temp_gz_filepath)

            print('done!')

    def _run_once_for_server_itsg(self):
        print('Connecting the remote server...')
        self._login()
        self._into_remote_filepath()

        files_list = self._ftp.nlst()
        files_list.sort()

        for i in range(len(files_list)):
            this_file = files_list[i]

            formatted_filename = self._format_filename(filename=this_file)
            pass
            if formatted_filename['product_type'] != self.configuration.product_type.name:
                continue
            if formatted_filename['beginning_date'] < self.configuration.beginning_date:
                continue
            if formatted_filename['ending_date'] > self.configuration.ending_date:
                continue
            if 'ITSG' + formatted_filename['product_release'] != self.configuration.release.name:
                continue

            local_l2path = self._get_local_l2path_from_filename(this_file)
            if self.configuration.update_mode and Path.exists(local_l2path):
                print(f'File {local_l2path.name} already exists at {local_l2path.parent}.')
                continue

            if not Path.exists(local_l2path.parent):
                Path.mkdir(local_l2path.parent, parents=True)

            with open(local_l2path, "wb") as f:
                file_handle = f.write

                print(f'Collecting {this_file}...', end=' ')
                self._ftp.retrbinary('RETR %s' % this_file, file_handle, blocksize=1024)

            print('done!')


def demo():
    collect = CollectL2SH()
    collect.configuration.set_server(L2DataServer.ITSG)
    collect.configuration.set_institute(L2InstituteType.ITSG)
    collect.configuration.set_release(L2Release.ITSGGrace2018)
    collect.configuration.set_degree(L2ProductMaxDegree.Degree96)
    collect.configuration.set_beginning_date(datetime.date(2010, 7, 1))
    collect.configuration.set_ending_date(datetime.date(2010, 7, 31))

    collect.run()


if __name__ == '__main__':
    demo()
