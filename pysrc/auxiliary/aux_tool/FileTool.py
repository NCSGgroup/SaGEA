from pathlib import Path
import gzip
import re

from pysrc.auxiliary.preference.EnumClasses import L2ProductType, L2InstituteType, L2Release, L2LowDegreeFileID
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool


class FileTool:
    @staticmethod
    def get_project_dir(sub=None, *, relative=False):
        dir_of_project = Path().absolute()
        relative_dir_str = Path('')

        i = 0
        while True:
            i += 1
            if i > 100:
                raise Exception

            dir_of_project = dir_of_project.parent
            # relative_dir_str += '../'
            relative_dir_str /= '..'
            if Path.exists(dir_of_project / 'pysrc'):
                break

        if relative:
            result = relative_dir_str
            # return Path(relative_dir_str)

        else:
            result = dir_of_project
            # return dir_of_project

        if sub is not None:
            result /= sub

        return result

    @staticmethod
    def get_l2_SH_path(year: int, month: int,
                       product_type: L2ProductType = L2ProductType.GSM,
                       institute: L2InstituteType = L2InstituteType.CSR,
                       release: L2Release = L2Release.RL06,
                       lmax: int = 60):
        """

        :param year:
        :param month:
        :param product_type:
        :param institute:
        :param release:
        :param lmax:
        :return: pathlib.Path, absolute path of GRACE L2 file
        """
        dir_up_to_year = FileTool.get_l2_SH_dir_upto_year(year, product_type, institute, release, lmax)

        filelist_this_year = list(Path.iterdir(dir_up_to_year))
        for i in range(len(filelist_this_year)):
            this_filename = filelist_this_year[i].name
            year_day_pattern = r'[A-Z]{3}-2_(\d{7})-(\d{7})'
            beginning_year_day, ending_year_day = re.search(year_day_pattern, this_filename).groups()

            beginning_date = TimeTool.convert_date_format(beginning_year_day, input_type=TimeTool.DateFormat.YearDay,
                                                          output_type=TimeTool.DateFormat.ClassDate)

            ending_date = TimeTool.convert_date_format(ending_year_day, input_type=TimeTool.DateFormat.YearDay,
                                                       output_type=TimeTool.DateFormat.ClassDate)

            ave_date = beginning_date + (ending_date - beginning_date) / 2

            if ave_date.month == month:
                return filelist_this_year[i]

        raise Exception('There is no local file of such time.')

    @staticmethod
    def get_l2_SH_dir_upto_year(year: int,
                                product_type: L2ProductType = L2ProductType.GSM,
                                institute: L2InstituteType = L2InstituteType.CSR,
                                release: L2Release = L2Release.RL06,
                                lmax: int = 60):

        if release == L2Release.RL06:
            assert lmax in (60, 96)
        if product_type == L2ProductType.GSM:
            degree_id_name = ('BA01', 'BB01')[(60, 96).index(lmax)]
        else:
            degree_id_name = 'BC01'

        dir_up_to_year = FileTool.get_project_dir() / 'data/L2_SH_Products/'
        dir_up_to_year /= f'{product_type.name}/{institute.name}/{release.name}/{degree_id_name}/{str(year)}'

        return dir_up_to_year

    @staticmethod
    def un_gz(gz_file_path: Path, target_path: Path = None):

        if target_path is None:
            target_path = gz_file_path.parent / gz_file_path.name.replace('.gz', '')

        g_file = gzip.GzipFile(gz_file_path)
        open(target_path, "wb+").write(g_file.read())
        g_file.close()

    @staticmethod
    def get_l2_low_deg_path(file_id: L2LowDegreeFileID,
                            institute: L2InstituteType = None,
                            release: L2Release = None):

        filepath = FileTool.get_project_dir() / 'data/L2_low_degrees'

        if file_id == L2LowDegreeFileID.TN11:
            filepath /= 'TN-11_C20_SLR_RL06.txt'

        elif file_id == L2LowDegreeFileID.TN13:
            assert institute in (L2InstituteType.CSR, L2InstituteType.GFZ, L2InstituteType.JPL)
            assert release in (L2Release.RL06, L2Release.RL061)

            institute_str = institute.name
            release_str = release.name.replace('RL061', 'RL06.1')
            filepath /= f'TN-13_GEOC_{institute_str}_{release_str}.txt'

        elif file_id == L2LowDegreeFileID.TN14:
            filepath /= 'TN-14_C30_C20_SLR_GSFC.txt'

        else:
            raise Exception

        return filepath
