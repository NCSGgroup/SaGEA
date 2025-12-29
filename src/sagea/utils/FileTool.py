from pathlib import Path
import gzip
import zipfile

import h5py


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
    def get_files_in_dir(fp, sub=False):
        assert fp.is_dir, f"{fp} is not a directory."

        file_list = []

        iterlist = list(fp.iterdir())
        for i in range(len(iterlist)):
            if iterlist[i].is_file():
                if not iterlist[i].name.startswith('.'):
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
