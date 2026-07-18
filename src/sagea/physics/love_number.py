from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as scio
from scipy import interpolate
from importlib.resources import files

from sagea.constants.constant import LoveNumberMethod


@dataclass
class LoveNumberConfig:
    method: LoveNumberMethod = LoveNumberMethod.Wang
    lmax: int = 60

    def set_lmax(self, lmax: int):
        self.lmax = int(lmax)
        return self

    def set_method(self, method: LoveNumberMethod):
        self.method = method
        return self


class LoveNumber:
    """
    Load load Love number k_l.

    Notes
    -----
    Returned array starts from degree 0.

    result[l] = k_l
    """

    def __init__(self):
        self.configuration = LoveNumberConfig()

    def config(self, config: LoveNumberConfig):
        self.configuration = config
        return self

    def get_Love_number(self) -> np.ndarray:
        method = self.configuration.method

        if method in (LoveNumberMethod.PREM, LoveNumberMethod.PREM.name):
            return self.__prem()

        if method in (LoveNumberMethod.AOD04, LoveNumberMethod.AOD04.name):
            return self.__aod04()

        if method in (LoveNumberMethod.Wang, LoveNumberMethod.Wang.name):
            return self.__wang()

        if method in (LoveNumberMethod.IERS, LoveNumberMethod.IERS.name):
            return self.__iers()

        raise ValueError(f"Unsupported Love number method: {method}")

    def __prem(self) -> np.ndarray:
        lmax = self.configuration.lmax

        if lmax >= 200:
            raise ValueError("PREM Love number is only reliable for lmax < 200.")

        index = np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50, 70, 100, 150, 200]
        )

        values = np.array(
            [
                0.000,
                0.027,
                -0.303,
                -0.194,
                -0.132,
                -0.104,
                -0.089,
                -0.081,
                -0.076,
                -0.072,
                -0.069,
                -0.064,
                -0.058,
                -0.051,
                -0.040,
                -0.033,
                -0.027,
                -0.020,
                -0.014,
                -0.010,
                -0.007,
            ]
        )

        degree = np.arange(lmax + 1)

        f = interpolate.interp1d(index, values, kind="cubic")
        return f(degree)

    def __aod04(self) -> np.ndarray:
        lmax = self.configuration.lmax

        if lmax > 100:
            raise ValueError("AOD04 Love number supports lmax <= 100.")

        k = np.zeros(101)

        k[0] = 0
        k[1] = 0
        k[2] = -0.308
        k[3] = -0.195
        k[4] = -0.132
        k[5] = -0.103
        k[6] = -0.089
        k[7] = -0.082
        k[8] = -0.078
        k[9] = -0.073

        for i in range(10, 18):
            k[i] = -(0.682 + 0.27 * (i - 10) / 8) / i

        for i in range(18, 32):
            k[i] = -(0.952 + 0.288 * (i - 18) / 14) / i

        for i in range(32, 56):
            k[i] = -(1.24 + 0.162 * (i - 32) / 24) / i

        for i in range(56, 101):
            k[i] = -(1.402 + 0.059 * (i - 56) / 44) / i

        return k[: lmax + 1]

    def __wang(self) -> np.ndarray:
        lmax = self.configuration.lmax

        if lmax > 360:
            raise ValueError("Wang Love number supports lmax <= 360.")

        if lmax == 0:
            return np.zeros(1)

        path = self.__get_auxiliary_file("LoveNumber.mat")

        love = scio.loadmat(path)
        mat = love["love"]

        # 原始文件通常从 degree 1 开始，因此取前 lmax 行并在 degree 0 补 0。
        kl_1_to_lmax = mat[:lmax, 3]

        return np.concatenate([[0.0], kl_1_to_lmax])

    def __iers(self) -> np.ndarray:
        lmax = self.configuration.lmax

        if lmax > 30:
            raise ValueError("IERS Love number supports lmax <= 30.")

        kl_1_to_30 = np.array(
            [
                0.000000000000000,
                -0.30750000000000,
                -0.19500000000000,
                -0.13200000000000,
                -0.10320000000000,
                -0.89166666666670e-1,
                -0.81710392640550e-1,
                -0.75500000000000e-1,
                -0.71685683412260e-1,
                -0.68200000000000e-1,
                -0.65980069344540e-1,
                -0.63812455645590e-1,
                -0.61732085548940e-1,
                -0.59754188127910e-1,
                -0.57883368816860e-1,
                -0.56118520212550e-1,
                -0.54455544917280e-1,
                -0.52888888888890e-1,
                -0.51529657180340e-1,
                -0.50236923831480e-1,
                -0.49007643741670e-1,
                -0.47838465083770e-1,
                -0.46725942423010e-1,
                -0.45666666666670e-1,
                -0.44657342166760e-1,
                -0.43694830109180e-1,
                -0.42776170404080e-1,
                -0.41898589949110e-1,
                -0.41059502372580e-1,
                -0.40256502584650e-1,
            ]
        )

        return np.concatenate([[0.0], kl_1_to_30[:lmax]])

    @staticmethod
    def __get_auxiliary_file(filename: str) -> Path:
        """
        Read package internal data.

        Expected path:
            src/sagea/data/auxiliary/LoveNumber.mat
        """

        return Path(str(files("sagea").joinpath("data", "auxiliary", filename)))
