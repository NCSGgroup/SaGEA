import ftplib
import re
from pathlib import Path

from pysrc.auxiliary.tools.FileTool import FileTool

from pysrc.auxiliary.preference.EnumClasses import L2DataServer, L2LowDegreeFileID


class CollectL2LowDegConfig:
    def __init__(self):
        self.server = L2DataServer.GFZ
        self.file_id = L2LowDegreeFileID.TN13

    def set_server(self, server: L2DataServer):
        self.server = server
        return self

    def set_file_id(self, file_id: L2LowDegreeFileID):
        """
        TN-11_C20_SLR_RL06.txt
        TN-13_GEOC_(CSR/GFZ/JPL)_(RL06/RL06.1).txt
        TN-14_C30_C20_SLR_GSFC.txt
        """
        self.file_id = file_id
        return self


class CollectL2LowDeg:
    def __init__(self):
        self.configuration = CollectL2LowDegConfig()
        pass

    def config(self, config: CollectL2LowDegConfig):
        self.configuration = config

        return self

    def run(self):
        if self.configuration.server == L2DataServer.GFZ:
            self._run_once_for_server_gfz()

        else:
            raise Exception

        return self

    def _get_hostname(self):
        if self.configuration.server == L2DataServer.GFZ:
            hostname = 'isdcftp.gfz-potsdam.de'

        else:
            raise Exception

        return hostname

    def _login(self):
        hostname = self._get_hostname()

        self._ftp = ftplib.FTP(hostname)
        self._ftp.login()

        return self

    def _into_remote_filepath(self):
        path = self._get_remote_low_deg_dir()
        path_str = str(path).replace('\\', '/')

        self._ftp.cwd(path_str)

        return self

    def _get_remote_low_deg_dir(self):
        if self.configuration.server == L2DataServer.GFZ:
            path = Path('grace/DOCUMENTS/TECHNICAL_NOTES')
        else:
            raise Exception

        return path

    def _get_file_id(self, filename):
        regular_pattern = r'.*TN-(\d{2}).*'
        match = re.match(regular_pattern, filename).groups()[0]

        file_id_str = f'TN{match}'

        file_id = None
        if file_id_str in L2LowDegreeFileID.__members__.keys():
            file_id = L2LowDegreeFileID.__members__[file_id_str]

        return file_id

    def _run_once_for_server_gfz(self):
        self._login()
        self._into_remote_filepath()

        files_list = self._ftp.nlst()
        files_list.sort()

        for i in range(len(files_list)):
            this_file = files_list[i]

            this_file_id = self._get_file_id(this_file)
            if this_file_id == self.configuration.file_id:
                local_low_deg_path = FileTool.get_project_dir() / f'data/L2_low_degrees/{this_file}'
            else:
                continue

            with open(local_low_deg_path, "wb") as f:
                file_handle = f.write

                print(f'Collecting {this_file}...', end=' ')
                self._ftp.retrbinary('RETR %s' % this_file, file_handle, blocksize=1024)
                print('done!')
