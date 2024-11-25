import os
import random
import shutil
import string
from pathlib import Path
import gzip
import zipfile
import re

import h5py

from pysrc.auxiliary.preference.EnumClasses import L2ProductType, L2InstituteType, L2Release, L2LowDegreeFileID
import pysrc.auxiliary.preference.EnumClasses as Enum
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

            if Path.exists(dir_of_project / 'pysrc'):
                break

            dir_of_project = dir_of_project.parent
            relative_dir_str /= '..'

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

            if institute in (L2InstituteType.CSR, L2InstituteType.GFZ, L2InstituteType.JPL):
                degree_id_name = ('BA01', 'BB01')[(60, 96).index(lmax)]
            elif institute == L2InstituteType.ITSG:
                degree_id_name = f'n{str(lmax)}'
            else:
                assert False
        else:
            degree_id_name = 'BC01'

        dir_up_to_year = FileTool.get_project_dir() / 'data/L2_SH_Products/'
        dir_up_to_year /= product_type.name
        dir_up_to_year /= institute.name
        dir_up_to_year /= release.name.replace('ITSG', '')
        dir_up_to_year /= degree_id_name
        dir_up_to_year /= str(year)

        return dir_up_to_year

    @staticmethod
    def get_files_in_dir(fp, sub=False):
        assert fp.is_dir, f"{fp} is not a directory."

        file_list = []

        iterlist = list(fp.iterdir())
        for i in range(len(iterlist)):
            if iterlist[i].is_file():
                file_list.append(iterlist[i])

            elif sub:
                file_list += FileTool.get_files_in_dir(iterlist[i], sub=sub)

        return file_list

    @staticmethod
    def un_gz(gz_file_path: Path, target_path: Path = None):

        if target_path is None:
            target_path = gz_file_path.parent / gz_file_path.name.replace('.gz', '')

        g_file = gzip.GzipFile(gz_file_path)
        open(target_path, "wb+").write(g_file.read())
        g_file.close()

    @staticmethod
    def un_zip(zip_file_path: Path, target_path: Path = None):
        if target_path is None:
            target_path = zip_file_path.parent / zip_file_path.name.replace('.zip', '')

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_path)

    @staticmethod
    def get_l2_low_deg_path(filedir: Path = None,
                            file_id: L2LowDegreeFileID = None,
                            institute: L2InstituteType = None,
                            release: L2Release = None
                            ):

        filedir = FileTool.get_project_dir() / 'data/L2_low_degrees' if filedir is None else filedir

        if file_id == L2LowDegreeFileID.TN11:
            filedir /= 'TN-11_C20_SLR_RL06.txt'

        elif file_id == L2LowDegreeFileID.TN13:
            assert institute in (L2InstituteType.CSR, L2InstituteType.GFZ, L2InstituteType.JPL)
            assert release in (L2Release.RL06, L2Release.RL061)

            institute_str = institute.name
            release_str = release.name.replace('RL061', 'RL06.1')
            filedir /= f'TN-13_GEOC_{institute_str}_{release_str}.txt'

        elif file_id == L2LowDegreeFileID.TN14:
            filedir /= 'TN-14_C30_C20_SLR_GSFC.txt'

        else:
            raise Exception

        return filedir

    @staticmethod
    def get_hdf5_structure(filepath):
        def append_structure(fdata, flevel=0, texts=None):
            if texts is None:
                texts = []

            texts.append(f"{'|  ' * flevel}|--{fdata.name.split('/')[-1]}")

            if type(fdata) is h5py._hl.group.Group:
                flevel += 1
                texts.append('|  ' * flevel + '|')
                for fkey in fdata.keys():
                    append_structure(fdata[fkey], flevel, texts)
                flevel -= 1
                texts.append('|  ' * flevel + '|')

            elif type(fdata) is h5py._hl.dataset.Dataset:
                lines[-1] += f' {fdata.shape}'
                pass

            return texts

        filepath = Path(filepath)
        assert filepath.name.endswith('.hdf5')
        lines = []

        with h5py.File(filepath, 'r') as f:
            lines.append(f'{filepath.name}')
            lines.append('|')

            for key in f.keys():
                lines = append_structure(f[key], texts=lines)

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].replace('|', '').replace(' ', '') == '':
                lines.pop(i)
            else:
                break

        return '\n'.join(lines)

    @staticmethod
    def get_GIA_path(filedir: Path = None, gia_type=Enum.GIAModel):
        assert gia_type in Enum.GIAModel

        filedir = FileTool.get_project_dir("data/GIA/") if filedir is None else filedir

        if gia_type == Enum.GIAModel.Caron2018:
            return filedir / "GIA.Caron_et_al_2018.txt"
        elif gia_type == Enum.GIAModel.Caron2019:
            return filedir / "GIA.Caron_Ivins_2019.txt"
        elif gia_type == Enum.GIAModel.ICE6GC:
            return filedir / "GIA.ICE-6G_C.txt"
        elif gia_type == Enum.GIAModel.ICE6GD:
            return filedir / "GIA.ICE-6G_D.txt"
        else:
            assert False

    @staticmethod
    def move_folder(src_folder, dst_folder):
        try:
            shutil.move(src_folder, dst_folder)
        except Exception as e:
            print(f"move files failed: {e}")

    @staticmethod
    def remove_file(filepath):
        os.remove(filepath)

    @staticmethod
    def add_ramdom_suffix(filename, length=None):
        if length is None:
            length = 16

        assert type(filename) is str or issubclass(type(filename), Path)

        random_str = ''.join(random.sample(string.ascii_letters + string.digits, length - 1)) + "_"

        if type(filename) is str:
            filename_split = filename.split('/')
            if len(filename_split) >= 2:
                return "/".join(filename_split[:-1]) + "/" + random_str + filename_split[-1]
            else:
                return random_str + filename_split[-1]

        elif issubclass(type(filename), Path):
            parent = filename.parent
            name = filename.name

            name_random = random_str + name

            return parent / name_random
