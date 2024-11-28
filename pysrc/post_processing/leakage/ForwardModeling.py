import copy

import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
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
        self.__observed_gqij = None

        self.__log = False

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

    def set_observed_grid(self, gqij: np.ndarray):
        self.__observed_gqij = gqij
        return self

    def get_observed_grid(self):
        return self.__observed_gqij

    def set_filter(self, cs_filter: SHCFilter):
        self.__filter = cs_filter
        return self

    def get_filter(self):
        return self.__filter

    def set_harmonic(self, harmonic: Harmonic):
        self.__harmonic = harmonic
        return self

    def get_harmonic(self):
        return self.__harmonic

    def set_basin_conservation(self, basin: np.ndarray):
        self.__basin_to_maintain_global_conservation = basin

        return self

    def get_basin_conservation(self):
        return self.__basin_to_maintain_global_conservation

    def set_basin(self, basin: np.ndarray):
        self.__basin = basin

        return self

    def get_basin(self):
        return self.__basin

    def set_print_log(self, log=True):
        self.__log = log
        return self

    def get_print_log(self):
        return self.__log


class ForwardModeling(Leakage):
    def __init__(self):
        super().__init__()

        self.configuration = ForwardModelingConfig()

    def apply_to(self, gqij: np.ndarray, get_grid=False):
        if self.configuration.get_observed_grid() is not None:
            observed_model = self.configuration.get_observed_grid()
        else:
            observed_model = copy.deepcopy(gqij)

        basin = self.configuration.get_basin()

        basin_to_conservation = self.configuration.get_basin_conservation()
        shc_filter = self.configuration.get_filter()
        har = self.configuration.get_harmonic()
        acceleration_factor = self.configuration.get_acceleration_factor()
        max_iter_times = self.configuration.get_max_iteration()

        true_model = keep_signals_in_basin(gqij, basin, basin_to_conservation)
        iter_times = 0
        print_log = self.configuration.get_print_log()
        while True:
            iter_times += 1
            if print_log:
                print(f'\rforward modeling: iter {iter_times}...', end='')

            cqlm, sqlm = har.analysis(true_model)
            cqlm_filtered, sqlm_filtered = shc_filter.apply_to(cqlm, sqlm)

            grids_predicted = har.synthesis(cqlm_filtered, sqlm_filtered)

            grids_difference = (observed_model - grids_predicted) * basin

            true_model += grids_difference * acceleration_factor
            true_model = keep_signals_in_basin(true_model, basin, basin_to_conservation)

            if iter_times >= max_iter_times:
                break
        print()

        if get_grid:
            return true_model

        else:
            return MathTool.global_integral(true_model * basin)

    def format(self):
        return 'Forward modeling'


def demo1():
    import time
    from pysrc.auxiliary.load_file.LoadL2SH import load_SHC
    from pysrc.auxiliary.aux_tool.FileTool import FileTool

    '''load shc'''
    multi_times = 220
    lmax = 60
    spatial_resolution = 1

    cqlm, sqlm = load_SHC(
        FileTool.get_project_dir('data/auxiliary/GIF48.gfc'),
        key='gfc',
        lmax=lmax,
        lmcs_in_queue=(2, 3, 4, 5)
    ).get_cs2d()

    cqlm, sqlm = np.array([cqlm[0]] * multi_times), np.array([sqlm[0]] * multi_times)

    basin_path = FileTool.get_project_dir('data/basin_mask/Amazon_maskSH.dat')
    basin_conservation_path = FileTool.get_project_dir('data/basin_mask/Ocean_maskSH.dat')

    paths = [basin_path, basin_conservation_path]
    basin_shc = load_SHC(*paths, key="", lmax=lmax, lmcs_in_queue=(1, 2, 3, 4))
    basin_grid = basin_shc.to_grid(grid_space=spatial_resolution)

    basins = basin_grid.value

    gs = Gaussian()
    gs.configuration.set_lmax(lmax)
    gs.configuration.set_filtering_radius(300)

    lat, lon = MathTool.get_global_lat_lon_range(spatial_resolution)
    har = Harmonic(lat, lon, lmax=lmax, option=1)

    gqij = har.synthesis(cqlm, sqlm)

    fm = ForwardModeling()
    fm.configuration.set_harmonic(har)
    fm.configuration.set_basin(basins[0])
    fm.configuration.set_basin_conservation(basins[1])
    fm.configuration.set_filter(gs)

    time1 = time.time()
    fm.apply_to(gqij)
    time2 = time.time()

    return time2 - time1


if __name__ == '__main__':
    t = demo1()
    print(t)
