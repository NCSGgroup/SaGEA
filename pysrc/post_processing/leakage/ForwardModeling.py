import copy
import pathlib

import numpy as np

from pysrc.data_class.DataClass import GRID
from pysrc.data_class.DataClass import SHC
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
from pysrc.post_processing.filter.Base import SHCFilter
from pysrc.post_processing.filter.Gaussian import Gaussian
from pysrc.post_processing.harmonic.Harmonic import Harmonic
from pysrc.post_processing.leakage.Base import Leakage


def keep_signals_in_basin(signals, basin, basin_to_maintain_global_conservation):
    new_signals = signals * basin
    total_signal = MathTool.global_integral(new_signals)

    basin_fixed = total_signal / MathTool.get_acreage(basin_to_maintain_global_conservation)

    new_signals -= np.einsum('i,jk->ijk', basin_fixed, basin_to_maintain_global_conservation)

    return new_signals


class ForwardModelingConfig:
    def __init__(self):
        self.__acceleration_factor = 1
        self.__max_iteration = 50
        self.__basin_to_maintain_global_conservation = None

        self.__basin = None
        self.__harmonic = None
        self.__filter = None

        self.__initial_grid = None
        self.__observed_grid = None
        self.__reverse_basin_mode = False

    def set_acceleration_factor(self, factor):
        self.__acceleration_factor = factor
        return self

    def get_acceleration_factor(self):
        return self.__acceleration_factor

    def set_max_iteration(self, max_iteration):
        self.__max_iteration = max_iteration
        return self

    def get_max_iteration(self):
        return self.__max_iteration

    def set_observed_grid(self, grid: GRID):
        self.__observed_grid = grid
        return self

    def get_observed_grid(self):
        return self.__observed_grid

    def set_filter(self, shc_filter: SHCFilter):
        self.__filter = shc_filter
        return self

    def get_filter(self):
        return self.__filter

    def set_harmonic(self, harmonic: Harmonic):
        self.__harmonic = harmonic
        return self

    def get_harmonic(self):
        return self.__harmonic

    def set_basin_conservation(self, basin: SHC or pathlib.WindowsPath):
        assert self.__harmonic is not None, "set harmonic before setting basin."

        har = self.__harmonic

        if type(basin) is pathlib.WindowsPath:
            lmax = self.__harmonic.lmax
            basin_clm, basin_slm = load_SHC(basin, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
            basin = har.synthesis(SHC(basin_clm, basin_slm)).value[0]
            basin[np.where(basin >= 0.5)] = 1
            basin[np.where(basin < 0.5)] = 0
            self.__basin_to_maintain_global_conservation = basin

        else:
            self.__basin_to_maintain_global_conservation = har.synthesis(basin).value[0]

        return self

    def get_basin_conservation(self):
        return self.__basin_to_maintain_global_conservation

    def set_basin(self, basin: SHC or pathlib.WindowsPath):
        assert self.__harmonic is not None, "set harmonic before setting basin."

        har = self.__harmonic

        if type(basin) is pathlib.WindowsPath:
            lmax = self.__harmonic.lmax
            basin_clm, basin_slm = load_SHC(basin, key='', lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
            basin = har.synthesis(SHC(basin_clm, basin_slm)).value[0]
            basin[np.where(basin >= 0.5)] = 1
            basin[np.where(basin < 0.5)] = 0
            self.__basin = basin

        elif type(basin) is SHC:
            self.__basin = har.synthesis(basin).value[0]

        elif type(basin) is np.ndarray:
            self.__basin = basin

        else:
            print('Unsupported format of basin map.')
            return -1

        return self

    def get_basin(self):
        return self.__basin

    def set_reverse_basin_mode(self, mode: bool):
        self.__reverse_basin_mode = mode
        return self

    def get_reverse_basin_mode(self):
        return self.__reverse_basin_mode


class ForwardModeling(Leakage):
    def __init__(self):
        super().__init__()

        self.configuration = ForwardModelingConfig()

    def apply_to(self, grid: GRID, get_grid=False):
        if self.configuration.get_observed_grid() is not None:
            observed_model = self.configuration.get_observed_grid().value
        else:
            observed_model = copy.deepcopy(grid.value)

        basin = self.configuration.get_basin()
        reverse_basin_mode = self.configuration.get_reverse_basin_mode()
        if reverse_basin_mode:
            basin = 1 - basin

        basin_to_conservation = self.configuration.get_basin_conservation()
        shc_filter = self.configuration.get_filter()
        har = self.configuration.get_harmonic()
        acceleration_factor = self.configuration.get_acceleration_factor()
        max_iter_times = self.configuration.get_max_iteration()

        true_model = keep_signals_in_basin(grid.value, basin, basin_to_conservation)
        iter_times = 0
        while True:
            iter_times += 1
            print(f'\rforward modeling: iter {iter_times}...', end='')

            cqlm, sqlm = har.analysis_for_gqij(true_model)
            cqlm_filtered, sqlm_filtered = shc_filter.apply_to(SHC(cqlm, sqlm)).get_cs2d()

            grids_predicted = har.synthesis_for_csqlm(cqlm_filtered, sqlm_filtered)

            grids_difference = (observed_model - grids_predicted) * basin

            true_model += grids_difference * acceleration_factor
            # true_model = keep_signals_in_basin(true_model, basin, basin_to_conservation)

            if iter_times >= max_iter_times:
                break
        print()

        if get_grid:
            grid = GRID(true_model, har.lat, har.lon)
            return grid

        else:
            if reverse_basin_mode:
                return - MathTool.global_integral(true_model * basin)
            else:
                return MathTool.global_integral(true_model * basin)

    def format(self):
        return 'Forward modeling'


def demo1():
    """synthesis/analysis for once"""
    import time
    from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
    from pysrc.auxiliary.aux_tool.FileTool import FileTool

    '''load shc'''
    multi_times = 220
    lmax = 60
    spatial_resolution = 1

    clm, slm = load_SHC(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    )
    cqlm, sqlm = np.array([clm] * multi_times), np.array([slm] * multi_times)
    shc = SHC(cqlm, sqlm)

    gs = Gaussian()
    gs.configuration.set_lmax(lmax)
    gs.configuration.set_filtering_radius(300)

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    grid = har.synthesis(shc)

    fm = ForwardModeling()
    fm.configuration.set_harmonic(har)
    fm.configuration.set_basin(FileTool.get_project_dir('data/basin_mask/Amazon_maskSH.dat'))
    fm.configuration.set_basin_conservation(FileTool.get_project_dir('data/basin_mask/Ocean_maskSH.dat'))
    fm.configuration.set_filter(gs)

    time1 = time.time()
    fm.apply_to(grid)
    time2 = time.time()

    return time2 - time1


if __name__ == '__main__':
    t = demo1()
    print(t)
