import itertools
from enum import Enum

import numpy as np
from scipy.optimize import minimize

from pysrc.auxiliary.preference.EnumClasses import match_string


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

    def get_variance(self):
        mode = self.configuration.get_mode()

        if mode == TCHMode.KKT:
            var_array = self.__run_with_KKT()
        elif mode == TCHMode.OLS:
            var_array = self.__run_with_OLS()
        else:
            assert False

        return var_array


def tch_estimate(*dataset, mode: TCHMode or str = TCHMode.OLS):
    if isinstance(mode, str):
        mode = match_string(mode, TCHMode, ignore_case=True)
    assert mode in TCHMode

    dataset = np.array(dataset)

    input_shape = dataset[0].shape

    tch = TCH()
    tch.configuration.set_mode(mode=mode)

    if len(input_shape) == 1:
        tch.set_datasets(*dataset)
        return tch.get_variance()

    else:
        nset = len(dataset)
        data_shape = input_shape[1:]
        data1d_length = np.prod(data_shape)

        data1d = [np.array([dataset[i][j].flatten() for j in range(len(dataset[i]))]) for i in range(nset)]

        tch_results_1d = np.zeros((nset, data1d_length))
        for i in range(data1d_length):
            tch.set_datasets(*[data1d[j][:, i] for j in range(nset)])
            tch_results_1d[:, i] = tch.get_variance()

        tch_results = tch_results_1d.reshape((nset, *data_shape))

        return tch_results
