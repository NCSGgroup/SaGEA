import itertools
from enum import Enum

import numpy as np
from scipy.optimize import minimize


class TCHMode(Enum):
    KKT = 1
    OLS = 2


class TCHConfig:
    def __init__(self):
        self.__mode = TCHMode.KKT

    def set_mode(self, mode: TCHMode):
        """
        :param mode: TCHMode

        if param mode is TCHMode.KKT, using Karush-Kuhn-Tucker Condition to solve, see ...;
        if param mode is TCHMode.OLS, using Ordinary Least Square to solve, see Chen. et al., 2019.
        """

        self.__mode = mode
        return self

    def get_mode(self):
        return self.__mode


class TCH:
    def __init__(self):
        self.configuration = TCHConfig()

        self.datasets = None
        self.xn = None  # number of datasets/institutes
        self.xm = None  # length of each dataset

        self.__r_mat = None

        pass

    def set_datasets(self, *x):
        """
        Number of datasets should be greater than or equal to 3, and the last one will be set as the preference.
        """
        self.xn = len(x)  # ...
        self.xm = len(x[0])  # ...
        assert self.xn >= 3  # ensure that ...

        self.datasets = np.array(x)

        return self

    def __run_with_KKT(self):
        x_mat = self.datasets.T  # M * N matrix: M denotes ...; N denotes ...
        x_mat -= np.mean(x_mat, axis=0)  # de-average

        y_mat = x_mat[:, :-1] - x_mat[:, -1][:, None]  # M * (N-1) matrix; the final N-index is the reference...
        y_mat -= np.mean(y_mat, axis=0)  # de-average

        s_mat = 1 / (self.xm - 1) * (y_mat.T @ y_mat)  # covariance matrx of y

        K = np.linalg.det(s_mat) ** (1 - self.xn)  # ...
        u = np.ones((self.xn - 1,))  # ...
        # s_mat_inv = np.linalg.inv(s_mat)
        s_mat_inv = np.linalg.pinv(s_mat)

        '''KKT to solve vector r[:, n]'''

        def objective(x):
            """length of 1-d array x should be equal to self.xn"""
            r_hat = s_mat - x[-1] * u[:, None] @ u[None, :] + u[:, None] @ x[None, :-1] + x[:-1, None] @ u[None, :]
            r_hat_tril = np.tril(r_hat, k=-1)

            return (np.sum(r_hat_tril ** 2) + np.sum(x[:-1] ** 2)) / (K ** 2)

        def constraint_ineq1(x):
            return -1 / K * (x[-1] - (x[:-1] - x[-1] * u).T @ s_mat_inv @ (x[:-1] - x[-1] * u))

        '''initial condition'''
        r_ini = np.concatenate([np.zeros((self.xn - 1,)), np.array([(2 * u.T @ s_mat_inv @ u) ** (-1)])])

        con1 = dict(type='ineq', fun=constraint_ineq1)
        solution = minimize(objective, r_ini, method='SLSQP', constraints=con1)

        r_with_rnn = solution.x  # ...
        r = r_with_rnn[:-1]
        rnn = r_with_rnn[-1]

        r_mat = np.zeros((self.xn, self.xn))  # ...

        r_mat[:-1, :-1] = s_mat - rnn * u[:, None] @ u[None, :] + u[:, None] @ r[None, :] + r[:, None] @ u[None, :]

        r_mat[-1, :-1] = r
        r_mat[:-1, -1] = r
        r_mat[-1, -1] = rnn

        var_array = np.abs(np.diag(r_mat))  # cov
        return var_array

    def __run_with_OLS(self):
        detasets_id = range(self.xn)
        combination_of_index = list(itertools.combinations(detasets_id, 2))
        length_of_combination = len(combination_of_index)

        A_mat = np.zeros((length_of_combination, self.xn))
        var_diff_array = np.zeros((length_of_combination, 1))

        for i in range(length_of_combination):
            A_mat[i, combination_of_index[i]] = 1

            diff_of_dataset = self.datasets[combination_of_index[i][0]] - self.datasets[combination_of_index[i][1]]

            var_diff_array[i] = np.var(diff_of_dataset)

        design_mat = np.linalg.inv(A_mat.T @ A_mat) @ A_mat.T
        var_array = np.abs(design_mat @ var_diff_array)[:, 0]
        return var_array

    def get_var_epsilon(self):
        mode = self.configuration.get_mode()

        if mode == TCHMode.KKT:
            var_array = self.__run_with_KKT()
        elif mode == TCHMode.OLS:
            var_array = self.__run_with_OLS()
        else:
            assert False

        return var_array


def demo():
    import datetime
    from tqdm import trange

    from pysrc.data_class.DataClass import SHC

    from pysrc.auxiliary.load_file.LoadL2SH import LoadL2SH, load_cs
    from pysrc.auxiliary.aux_tool.MathTool import MathTool
    from pysrc.auxiliary.aux_tool.FileTool import FileTool
    from pysrc.auxiliary.preference.EnumClasses import L2InstituteType

    from pysrc.post_processing.convert_field_physical_quantity.ConvertSHC import ConvertSHC
    from pysrc.post_processing.Love_number.LoveNumber import LoveNumber
    from pysrc.post_processing.harmonic.Harmonic import Harmonic
    from pysrc.post_processing.filter.Gaussian import Gaussian

    from pysrc.auxiliary.scripts.PlotGrids import plot_grids

    '''TCH config'''
    tch_mode = TCHMode.OLS

    '''load gsm'''
    gif48_path = FileTool.get_project_dir() / 'data/auxiliary/GIF48.gfc'
    clm_bg, slm_bg = load_cs(gif48_path, key='gfc', lmax=60)
    shc_bg = SHC(clm_bg, slm_bg)

    load = LoadL2SH()
    load.configuration.set_begin_date(datetime.date(2005, 1, 1))
    load.configuration.set_end_date(datetime.date(2015, 12, 31))

    load.configuration.set_institute(L2InstituteType.CSR)
    shc_csr = load.get_shc()
    shc_csr.de_background(shc_bg)

    load.configuration.set_institute(L2InstituteType.GFZ)
    shc_gfz = load.get_shc()
    shc_gfz.de_background(shc_bg)

    load.configuration.set_institute(L2InstituteType.JPL)
    shc_jpl = load.get_shc()
    shc_jpl.de_background(shc_bg)

    shc_csr.value[:, :6] = 0
    shc_gfz.value[:, :6] = 0
    shc_jpl.value[:, :6] = 0

    '''convert to ewh'''
    ln = LoveNumber().get_Love_number()
    convert = ConvertSHC()
    convert.configuration.set_Love_number(ln)
    shc_csr = convert.apply_to(shc_csr)
    shc_gfz = convert.apply_to(shc_gfz)
    shc_jpl = convert.apply_to(shc_jpl)

    '''gs filter'''
    gs = Gaussian()
    gs.configuration.set_filtering_radius(300)
    shc_csr = gs.apply_to(shc_csr)
    shc_gfz = gs.apply_to(shc_gfz)
    shc_jpl = gs.apply_to(shc_jpl)

    '''harmonic synthesis to gridded signal'''
    grid_space = 1
    lmax = 60
    lat, lon = MathTool.get_global_lat_lon_range(grid_space)
    har = Harmonic(lat, lon, lmax, option=1)
    grids_csr = har.synthesis(shc_csr)
    grids_gfz = har.synthesis(shc_gfz)
    grids_jpl = har.synthesis(shc_jpl)

    '''tch for grid'''
    grid_shape = np.shape(grids_csr.value)[-2:]
    grid_length = grid_shape[0] * grid_shape[1]

    grids_tch_1d = np.zeros((3, grid_length))
    tch = TCH()
    tch.configuration.set_mode(mode=tch_mode)

    grids_flatten_csr = np.array([grids_csr.value[i].flatten() for i in range(len(grids_csr.value))])
    grids_flatten_gfz = np.array([grids_gfz.value[i].flatten() for i in range(len(grids_gfz.value))])
    grids_flatten_jpl = np.array([grids_jpl.value[i].flatten() for i in range(len(grids_jpl.value))])

    for i in trange(grid_length):
        tch.set_datasets(grids_flatten_jpl[:, i], grids_flatten_gfz[:, i], grids_flatten_csr[:, i])
        grids_tch_1d[:, i] = tch.get_var_epsilon()

    grids_tch = np.sqrt(grids_tch_1d).reshape((3, *grid_shape))

    plot_grids(
        grids_tch * 100,  # cm
        lat=lat,
        lon=lon,
        vmin=0.,
        vmax=6.,
        subtitle=['JPL', 'GFZ', 'CSR'],
        title="TCH with OLS"
    )


if __name__ == '__main__':
    demo()
