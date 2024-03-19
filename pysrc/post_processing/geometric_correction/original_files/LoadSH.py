"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/10/28 下午4:56
@Description:
"""

import os
import warnings

import numpy as np


class LoadSH:
    """
    This class is a base class that deals with the gravity models reading.
    """

    def __init__(self):
        self.product_type = None
        self.modelname = None
        self.GM = None
        self.Radius = None
        self.maxDegree = None
        self.zero_tide = None
        self.errors = None
        self.norm = None
        self.date_begin = None
        self.date_end = None
        self.date_middle = None
        self.unusedDays = None

        self._sigmaC, self._sigmaS = None, None
        self._C, self._S = None, None
        pass

    def load(self, fileIn: str):
        return self

    def getCS(self, Nmax):

        assert Nmax >= 0

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        if end > len(self._C):
            warnings.warn('Nmax is too big')

        return self._C[0:end].copy(), self._S[0:end].copy()

    def getSigmaCS(self, Nmax):

        assert Nmax >= 0

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        if end > len(self._C):
            warnings.warn('Nmax is too big')

        end = int((Nmax + 1) * (Nmax + 2) / 2)

        return self._sigmaC[0:end].copy(), self._sigmaS[0:end].copy()

    def replace(self):
        """
        :return:
        """
        return self._C, self._S, self._sigmaC, self._sigmaS


class SimpleSH(LoadSH):
    """
    specified to the ones in the simplest format (No header) like below:

    0  0  c s
    1  0  c s
    1  1  c s
    """

    def __init__(self):
        LoadSH.__init__(self)

    def load(self, fileIn: str):

        with open(fileIn) as f:
            content = f.readlines()
            pass

        l, m, C, S = [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0: continue

            l.append(value[0])
            m.append(value[1])
            C.append(value[2])
            S.append(value[3])

        l = np.array(l).astype(int)
        m = np.array(m).astype(int)
        C = np.array(C).astype(float)
        S = np.array(S).astype(float)

        n = np.round(np.sqrt(len(l) * 2)) - 1
        assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(np.int)] = C
        self._S[(l * (l + 1) / 2 + m).astype(np.int)] = S

        return self


class Gif48(LoadSH):
    """
    specified to read gif48 gravity fields. Plus, the file that formats the same as GIF48 can be read as well.
    """

    def __init__(self):
        LoadSH.__init__(self)
        # self.__sigmaC, self.__sigmaS = None, None
        # self.__C, self.__S = None, None

    def load(self, fileIn: str):
        """
        load gif48 fields
        :param fileIn: gif48 file and its path
        :return:
        """

        flag = 0

        with open(fileIn) as f:
            content = f.readlines()
            pass

        flag = self.__header(content)

        self.__read(content[flag + 1:])

        return self

    def __header(self, content: list):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):
            value = content[i].split()
            if len(value) == 0: continue

            if value[0] == 'product_type':
                self.product_type = value[1]
            elif value[0] == 'modelname':
                self.modelname = value[1]
            elif value[0] == 'earth_gravity_constant':
                self.GM = float(value[1])
            elif value[0] == 'radius':
                self.Radius = float(value[1])
            elif value[0] == 'max_degree':
                self.maxDegree = int(value[1])
            elif value[0] == 'errors':
                self.errors = value[1]
            elif value[0] == 'norm':
                self.norm = value[1]
            elif value[0] == 'tide_system':
                self.zero_tide = (value[1] == 'zero_tide')
            elif value[0] == 'end_of_head':
                flag = i
                break

        return flag

    def __read(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0: continue

            l.append(value[1])
            m.append(value[2])
            C.append(value[3])
            S.append(value[4])
            sigmaC.append(value[5])
            sigmaS.append(value[6])

        l = np.array(l).astype(int)
        m = np.array(m).astype(int)
        C = np.array(C).astype(float)
        S = np.array(S).astype(float)
        sigmaC = np.array(sigmaC).astype(float)
        sigmaS = np.array(sigmaS).astype(float)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(int)] = C
        self._S[(l * (l + 1) / 2 + m).astype(int)] = S
        self._sigmaC[(l * (l + 1) / 2 + m).astype(int)] = sigmaC
        self._sigmaS[(l * (l + 1) / 2 + m).astype(int)] = sigmaS


class Level2Gravity(LoadSH):
    """
    This is used to read level 2 temporal gravity fields from official data centers: CSR, JPL and GFZ
    """

    def load(self, fileIn: str):
        """
        Read one file using given directory and filename. This function asks for an exact filename.
        :param fileIn:
        :return:
        """

        flag = 0

        with open(fileIn) as f:
            content = f.readlines()
            pass

        flag = self.__header(content)

        self.__read(content[flag + 1:])

        return self

    def __header(self, content: list):
        """
        Read and identify the header of the gif48 gravity fields file
        :param content: content of the file
        :return: the line of "end_of_header"
        """

        flag = 0

        for i in range(len(content)):
            value = content[i].split(':')

            if len(value) == 0: continue
            if 'End of YAML header' in value[0]:
                flag = i
                break

            if 'time_coverage_start' in value[0]:
                self.date_begin = value[1].strip().split('T')[0]
            if 'time_coverage_end' in value[0]:
                self.date_end = value[1].strip().split('T')[0]
            if 'unused_days' in value[0]:
                self.unusedDays = value[1].strip().strip()[1:-1].split(',')

            if 'title' in value[0]:
                self.product_type = value[1].strip()
            elif 'product_version' in value[0]:
                self.modelname = value[1].strip()
            elif 'mean_equator_radius' in value[0]:
                self.Radius = float(content[i + 3].split(':')[1])
            elif 'earth_gravity_param' in value[0]:
                self.GM = float(content[i + 3].split(':')[1])
            elif 'normalization' in value[0]:
                self.norm = value[1].strip()
            elif 'permanent_tide_flag' in value[0]:
                self.zero_tide = value[1].strip()
            elif 'dimensions' in value[0]:
                self.maxDegree = int(content[i + 1].split(':')[1])
            elif 'comments' in value[0]:
                self.errors = value[1].split()[0]

        '''compute the middle date in terms of month: for the later time-series analysis'''
        '''Time is not evenly distributed'''
        be = int(self.date_begin[:4]) + ((int(self.date_begin[5:7])-1)*(365.25/12)+ int(self.date_begin[8:10]))/365.25
        en= int(self.date_end[:4]) + (
                    (int(self.date_end[5:7]) - 1) * (365.25 / 12) + int(self.date_end[8:10])) / 365.25
        self.date_middle = (be+en)/2*12

        return flag

    def __read(self, content: list):
        """
        Start reading gif48 like line by line like below
        key    L    M    C    S    sigma C    sigma S
        :param content:
        :return: all the C, S coefficients in order
        """
        l, m, C, S, sigmaC, sigmaS = [], [], [], [], [], []

        for item in content:
            value = item.split()
            if len(value) == 0: continue

            l.append(value[1])
            m.append(value[2])
            C.append(value[3])
            S.append(value[4])
            sigmaC.append(value[5])
            sigmaS.append(value[6])

        l = np.array(l).astype(int)
        m = np.array(m).astype(int)
        C = np.array(C).astype(float)
        S = np.array(S).astype(float)
        sigmaC = np.array(sigmaC).astype(float)
        sigmaS = np.array(sigmaS).astype(float)

        if self.maxDegree is not None:
            assert len(l) == int((self.maxDegree + 1) * (self.maxDegree + 2) / 2)
        else:
            n = np.round(np.sqrt(len(l) * 2)) - 1
            assert len(l) == int((n + 1) * (n + 2) / 2)

        self._C = np.zeros(len(l))
        self._S = np.zeros(len(l))
        self._sigmaC = np.zeros(len(l))
        self._sigmaS = np.zeros(len(l))

        self._C[(l * (l + 1) / 2 + m).astype(int)] = C
        self._S[(l * (l + 1) / 2 + m).astype(int)] = S
        self._sigmaC[(l * (l + 1) / 2 + m).astype(int)] = sigmaC
        self._sigmaS[(l * (l + 1) / 2 + m).astype(int)] = sigmaS


def LoadGsmByYear(localDir: str, begin: str, end: str, opt='60'):
    """
    load files by years
    :param opt: degree/order = 60 or 90
    :param localDir: directory where temporal gravity fields are saved
    :param begin: e.g., '2009'
    :param end: e.g., '2010'
    :return: a list of time-series GSM sorted by date
    """
    res = []
    assert opt in ['60', '90']

    id = None
    if opt == '60':
        id = 'BA01'
    elif opt == '90':
        id = 'BB01'

    target = None
    years = np.arange(int(begin), int(end) + 1)

    for root, dirs, files in os.walk(localDir):
        for name in files:
            str_date = name.split('_')[1].split('-')
            start = str_date[0]

            if ('GSM' in name) and (id in name) and (int(start[:4]) in years):
                target = os.path.join(root, name)
                res.append(Level2Gravity().load(target))
                pass

    '''
    sort the list by date
    '''

    dates = [x.date_begin for x in res]
    index = sorted(range(len(dates)), key=lambda k: dates[k])
    ress = [res[i] for i in index]

    return ress


def demo1():
    l2sh = Level2Gravity().load('../data/L2_SH_products/CSR/GSM/BA01/2002/GSM-2_2002095-2002120_GRAC_UTCSR_BA01_0600')
    pass


def demo2():
    ts = LoadGsmByYear(localDir='../data/L2_SH_products/CSR', begin='2002', end='2020', opt='90')
    for el in ts:
        print(el.date_begin)
    pass


if __name__ == '__main__':
    demo2()
