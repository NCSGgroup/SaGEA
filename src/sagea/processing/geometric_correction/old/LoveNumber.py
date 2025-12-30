# @Time     : 2019/11/27  17:36
# @Author   : Yang Fan
# @Contact  : fanyang1990@foxmail.com 
# @Address  : Huazhong University of Science and Technology
# @File     : LoveNumber.py
# @Software : PyCharm
# @Function : This code is dedicated to ...

import numpy as np
import scipy.io as scio
from scipy import interpolate

from sagea.processing.geometric_correction.old.Setting import LoveNumberType


class LoveNumber:
    """
    Notice: all love numbers are given with starting index of 0
    """

    def __init__(self, fileDir: str):

        self.__path = fileDir
        self.__Nmax = None

        pass

    def getNumber(self, Nmax: int, method=LoveNumberType.AOD04):
        self.__Nmax = Nmax
        func = None
        if method == LoveNumberType.PREM:
            func = self.__PREM
        elif method == LoveNumberType.AOD04:
            func = self.__AOD04
        elif method == LoveNumberType.Wang:
            func = self.__Wang
        elif method == LoveNumberType.IERS:
            func = self.__IERS

        return func()

    def __PREM(self):
        """
        PREM model
        :return:
        """

        assert self.__Nmax < 200
        '''The code is unreliable once greater than 200'''

        index = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                          10, 12, 15, 20, 30, 40, 50,
                          70, 100, 150, 200])

        values = np.array([0.000, 0.027, -0.303, -0.194,
                           -0.132, -0.104, -0.089, -0.081,
                           -0.076, -0.072, -0.069, -0.064,
                           -0.058, -0.051, -0.040, -0.033,
                           -0.027, -0.020, -0.014, -0.010, -0.007])

        xnew = np.array(range(0, self.__Nmax + 1))

        f = interpolate.interp1d(index, values, kind="cubic")
        # ‘slinear', ‘quadratic' and ‘cubic' refer to a spline interpolation of first, second or third order)
        ynew = f(xnew)

        # print(ynew)

        return ynew

    def __AOD04(self):
        """
        Love number that was used in AOD RL04
        :return:
        """

        assert self.__Nmax <= 100

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

        # print(k[0:(self.__Nmax+1)])

        return k[0:(self.__Nmax + 1)]

    def __Wang(self):
        """
        This is a more accurate way to extract Love number, which is recommended.
        Starting from degree 0
        :return:
        """

        assert self.__Nmax <= 360

        path = str(self.__path) + '/LoveNumber'

        love = scio.loadmat(path)

        # love = scio.loadmat('raw_data/test/love')

        kl = love['love'][0:self.__Nmax, 3]

        # print(kl)

        return np.append(np.zeros(1), kl)

    def __IERS(self):
        """
        obtained from IERS2010 convention.
        :return:
        """

        assert self.__Nmax <= 30
        '''
        load love number, from 1 to 30
        '''
        Kn = np.array([
            0.000000000000000, -0.30750000000000,
            -0.19500000000000, -0.13200000000000,
            -0.10320000000000, -0.89166666666670e-1,
            -0.81710392640550e-1, -0.75500000000000e-1,
            -0.71685683412260e-1, -0.68200000000000e-1,
            -0.65980069344540e-1, -0.63812455645590e-1,
            -0.61732085548940e-1, -0.59754188127910e-1,
            -0.57883368816860e-1, -0.56118520212550e-1,
            -0.54455544917280e-1, -0.52888888888890e-1,
            -0.51529657180340e-1, -0.50236923831480e-1,
            -0.49007643741670e-1, -0.47838465083770e-1,
            -0.46725942423010e-1, -0.45666666666670e-1,
            -0.44657342166760e-1, -0.43694830109180e-1,
            -0.42776170404080e-1, -0.41898589949110e-1,
            -0.41059502372580e-1, -0.40256502584650e-1
        ])

        return np.append(np.zeros(1), Kn[0:self.__Nmax])


def demo():
    LN = LoveNumber('../data/auxiliary/')
    Kn = LN.getNumber(30, LoveNumberType.Wang)
    pass


if __name__ == '__main__':
    demo()