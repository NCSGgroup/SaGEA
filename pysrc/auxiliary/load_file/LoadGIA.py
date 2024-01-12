import pathlib
import warnings

import numpy as np

from pysrc.auxiliary.core.SHC import SHC
from pysrc.auxiliary.load_file.LoadL2SH import load_SH_simple
from pysrc.auxiliary.preference.EnumClasses import GIAModel
from pysrc.auxiliary.tools.FileTool import FileTool


class LoadGIAConfig:
    def __init__(self):
        self.__lmax: int = 60
        self.__filepath = None

    def set_filepath(self, filepath: pathlib.WindowsPath):
        warnings.warn('This function is deprecated, use .set_GIA_model instead', DeprecationWarning)

        self.__filepath = filepath
        return self

    def set_GIA_model(self, model: GIAModel):
        filename = None
        if model == GIAModel.Caron2018:
            filename = 'GIA.Caron_et_al_2018.txt'

        elif model == GIAModel.Caron2019:
            filename = 'GIA.Caron_Ivins_2019.txt'

        elif model == GIAModel.ICE6GC:
            filename = 'GIA.ICE-6G_C.txt'

        elif model == GIAModel.ICE6GD:
            filename = 'GIA.ICE-6G_D.txt'

        self.__filepath = FileTool.get_project_dir() / f'data/GIA/{filename}'

        return self

    def get_filepath(self):
        return self.__filepath

    def set_lmax(self, lmax: int):
        self.__lmax = lmax
        return self

    def get_lmax(self):
        return self.__lmax


class LoadGIA:
    def __init__(self):
        self.configuration = LoadGIAConfig()

    def get_shc(self):
        gia_filepath = self.configuration.get_filepath()
        lmax = self.configuration.get_lmax()

        clm, slm = load_SH_simple(gia_filepath, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
        shc = SHC(clm, slm)

        return shc


def demo():
    filepath = FileTool.get_project_dir() / 'data/GIA/GIA.ICE-6G_D.txt'
    lmax = 60

    load = LoadGIA()

    load.configuration.set_filepath(filepath)
    load.configuration.set_lmax(lmax)

    shc = load.get_shc()
    pass


if __name__ == '__main__':
    demo()
