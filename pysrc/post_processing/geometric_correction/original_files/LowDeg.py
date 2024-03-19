"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2020/10/29 上午9:46
@Description:
"""

import math
import os

import numpy as np

from pysrc.post_processing.geometric_correction.original_files.LoadSH import LoadSH


class LowDegreeReplace:

    def __init__(self):
        self.__grav = None
        self.__SLR = None
        self.__GFZ = None
        self.__CSR = None
        self.__JPL = None
        self.__degree_one = None
        self.__index = None
        pass

    def load(self, localDir: str):
        """
        load all necessary files
        :param localDir: directory where low-degree SHs are replaced
        :return:
        """
        self.__CSR = self.__readDegreeOne(os.path.join(localDir, 'TN-13_GEOC_CSR_RL06.txt'))
        self.__GFZ = self.__readDegreeOne(os.path.join(localDir, 'TN-13_GEOC_GFZ_RL06.txt'))
        self.__JPL = self.__readDegreeOne(os.path.join(localDir, 'TN-13_GEOC_JPL_RL06.txt'))
        self.__SLR = self.__readC20C30(os.path.join(localDir, 'TN-14_C30_C20_SLR_GSFC.txt'))

        return self

    def __readDegreeOne(self, fullpath: str):

        flag = 0

        with open(fullpath) as f:
            content = f.readlines()
            pass

        for i in range(len(content)):
            if 'end of header' in content[i]:
                flag = i
                break

        cslist = []

        for item in content[flag + 1:]:
            value = item.split()
            if len(value) == 0: continue

            if int(value[1]) == 1 and int(value[2]) == 0:
                kmap = {}
                kmap['10'] = [float(value[3]), float(value[4]), float(value[5]), float(value[6])]
            elif int(value[1]) == 1 and int(value[2]) == 1:
                kmap['11'] = [float(value[3]), float(value[4]), float(value[5]), float(value[6])]
                start = value[7][:8]
                end = value[8][:8]
                flag1 = int(start[:4]) + (
                            (int(start[4:6]) - 1) * (365.25 / 12) + int(start[6:8])) / 365.25
                flag2 = int(end[:4]) + (
                        (int(end[4:6]) - 1) * (365.25 / 12) + int(end[6:8])) / 365.25
                flag = (flag1 + flag2) / 2

                kmap['date'] = flag
                cslist.append(kmap.copy())

        return cslist, [x['date'] for x in cslist]

    def __readC20C30(self, fullpath: str):
        flag = 0

        with open(fullpath) as f:
            content = f.readlines()
            pass

        for i in range(len(content)):
            if 'Product:' in content[i]:
                flag = i
                break

        cslist = []

        for item in content[flag + 1:]:
            value = item.split()
            if len(value) == 0: continue
            kmap = {}
            kmap['20'] = [float(value[2]), float(value[4]) * 1e-10]
            kmap['30'] = [float(value[5]), float(value[7]) * 1e-10]
            kmap['date'] = 1/2*(float(value[1])+float(value[-1]))
            cslist.append(kmap.copy())

        return cslist, [x['date'] for x in cslist]

    def setGrav(self, Grav: LoadSH):
        """
        Configure the gravity fields to be processed
        :param Grav: specify the gravity field
        :return:
        """
        f1, f2 = None, None
        if 'CSR' in Grav.product_type:
            self.__degree_one = self.__CSR
        elif 'GFZ' in Grav.product_type:
            self.__degree_one = self.__GFZ
        elif 'JPL' in Grav.product_type:
            self.__degree_one = self.__JPL

        f1, f2 = self.__degree_one
        f3 = self.__SLR[1]
        '''
        look for the corresponding date in replaced coefficients
        '''

        # flag1 = int(Grav.date_begin[:4]) + ((int(Grav.date_begin[5:7])-1)*(365.25/12)+ int(Grav.date_begin[8:10]))/365.25
        # flag2 = int(Grav.date_end[:4]) + (
        #             (int(Grav.date_end[5:7]) - 1) * (365.25 / 12) + int(Grav.date_end[8:10])) / 365.25
        # flag = (flag1+flag2)/2

        # if np.fabs(flag - Grav.date_middle/12)>0.000001:
        #     print('error')
        #     return

        flag = Grav.date_middle/12

        xlist = [np.fabs(x-flag) for x in f2]
        ylist = [np.fabs(x-flag) for x in f3]

        self.__index = [xlist.index(min(xlist)), ylist.index(min(ylist))]
        print(Grav.date_begin, Grav.date_end)
        self.__grav = Grav
        return self

    def rmDegZero(self):
        """
        set the degree into zero
        :return:
        """
        C, S, sigC, sigS = self.__grav.replace()
        C[0] = 0.
        S[0] = 0.
        sigC[0] = 0.
        sigS[0] = 0.
        return self

    def rpDegOne(self):
        """
        set the degree one-term as the ones provided in Technique Report
        :return:
        """
        C, S, sigC, sigS = self.__grav.replace()
        element = self.__degree_one[0][self.__index[0]]

        C[1], C[2] = element['10'][0], element['11'][0]
        S[1], S[2] = element['10'][1], element['11'][1]
        sigC[1], sigC[2] = element['10'][2], element['11'][2]
        sigS[1], sigS[2] = element['10'][3], element['11'][3]

        return self

    def rpDegTwo(self):
        """
        replace C20 term with SLR result
        :return:
        """
        C, S, sigC, sigS = self.__grav.replace()
        element = self.__SLR[0][self.__index[1]]

        C[3] = element['20'][0]
        sigC[3] = element['20'][1]

        return self

    def rpDegThree(self):
        """
        This function performs a replacement of C30 term with SLR result.
        Notice: it should be careful to carry out C30 replacement as it was suggested by Loomis(2020) that only
        C30 after 2017 is necessary to be replaced.
        :return:
        """
        C, S, sigC, sigS = self.__grav.replace()
        element = self.__SLR[0][self.__index[1]]

        if math.isnan(element['30'][0]):
            print('C30 is not offered for this epoch')
            return self

        C[6] = element['30'][0]
        sigC[6] = element['30'][1]

        return self

