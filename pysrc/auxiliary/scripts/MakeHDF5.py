import pathlib

import h5py
import numpy as np

from pysrc.auxiliary.tools.FileTool import FileTool


def make_hdf5(filepath: pathlib.WindowsPath or str, v: dict, make_dir=True, rewrite=False):
    """
    make a hdf5 file from a given dict.
    :param filepath:
    :param v: a dict that can be nested, and its value must be np.ndarray array or a string.
    :param make_dir: if True, it will be automatically generate the parent directory when it does not exist.
    :param rewrite: if True, it will allow the newly generated file to overwrite the possibly existing file.
    """

    assert type(filepath) in (pathlib.WindowsPath, str)

    if type(filepath) in (str,):
        path = pathlib.Path(filepath)
    else:
        path = filepath

    assert path.name.endswith('.hdf5')

    if path.exists() and not rewrite:
        raise Exception(f'file {path} already exists.')

    elif not path.parent.exists() and make_dir:
        path.parent.mkdir()

    def __write_dict(hfile, d: dict):
        for key in d.keys():

            assert type(d[key]) in (int, float, bool, complex, str, np.ndarray, dict)

            if type(d[key]) is dict:
                group = hfile.create_group(key)
                __write_dict(group, d[key])

            else:
                if type(d[key]) is str:
                    hfile.create_dataset(key, data=d[key], dtype=h5py.special_dtype(vlen=str))
                else:
                    hfile.create_dataset(key, data=d[key])

    with h5py.File(path, 'w') as f:
        __write_dict(f, v)


def read_hdf5(filename):
    def __make_dict(hdf5file):
        result = dict()

        for key in hdf5file.keys():
            if type(hdf5file[key]) is h5py._hl.dataset.Dataset:
                data = np.array(hdf5file[key])
                if data.ndim == 0:
                    if type(data[()]) is bytes:
                        data_to_store = str(data[()], 'utf-8')
                    else:
                        data_to_store = data[()]
                else:
                    data_to_store = data

                result[key] = data_to_store

            elif type(hdf5file[key]) is h5py._hl.group.Group:
                result[key] = __make_dict(hdf5file[key])

        return result

    with h5py.File(filename, 'r') as f:
        d = __make_dict(f)

    return d


def demo():
    d = dict(
        a=np.array([1, 2, 3]),
        b='123',
        c=dict(
            c1=np.array([1, 2, 3]),
            c2=True,
            c3=dict(
                c31=np.array([1, 2, 3]),
                c32=123,
            )
        )
    )

    h5_filepath = FileTool.get_project_dir() / 'temp/20231109/make_h5_1.hdf5'

    # make_hdf5(filepath=h5_filepath, v=d, rewrite=True)

    dd = read_hdf5(h5_filepath)
    print(dd)


if __name__ == '__main__':
    demo()
