import pathlib
import re
from datetime import date

import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.auxiliary.aux_tool.TimeTool import TimeTool


class ReadSinex:
    def __init__(self):
        self.__X = None
        self.__L = None
        self.__N = None
        self.__X0 = None
        self.__degree = None
        self.__unknowns = None
        self.__observations = None
        self.__weightSquare = None
        self.variMatrix = None
        self.formalError = None
        self.maxN = None
        self.__C_index = None
        self.__S_index = None
        pass

    def readComment(self, snx):
        while True:
            data = snx.readline()
            if data.split()[0] == '-FILE/COMMENT':
                break
            if data.split()[0] == 'max_degree':
                self.__degree = int(data.split()[1])
            if data.split()[0] == 'NUMBER OF UNKNOWNS':
                self.__unknowns = int(data.split()[1])
        pass

    def readSTATISTICS(self, snx):
        while True:
            data = snx.readline()
            if data.split()[0] == '-SOLUTION/STATISTICS':
                break
            if len(data.split()) == 4 and data.split()[2] == 'UNKNOWNS':
                self.__unknowns = int(data.split()[3])
            if len(data.split()) == 4 and data.split()[2] == 'OBSERVATIONS':
                self.__observations = int(data.split()[3])
            if len(data.split()) == 6 and data.split()[0] == 'WEIGHTED':
                self.__weightSquare = float(data.split()[5])
        self.__C_index = np.zeros((self.__degree + 1, self.__degree + 1))
        self.__S_index = np.zeros((self.__degree + 1, self.__degree + 1))
        self.__N = np.zeros((self.__unknowns, self.__unknowns))
        self.__L = np.zeros((self.__unknowns, 1))
        self.__X = np.zeros((self.__unknowns, 1))
        self.__X0 = np.zeros((self.__unknowns, 1))
        pass

    def readESTIMATE(self, snx):
        while True:
            data = snx.readline()
            if data.split()[0] == '-SOLUTION/ESTIMATE':
                break
            if data.split()[0] == '*INDEX':
                continue
            data_list = data.split()
            # self.getIndex(data_list)
            index = int(data_list[0])
            value = float(data_list[8])
            self.__X[index - 1][0] = value
        pass

    def readAPRIORI(self, snx):
        while True:
            data = snx.readline()
            if data.split()[0] == '-SOLUTION/APRIORI':
                break
            if data.split()[0] == '*INDEX':
                continue
            data_list = data.split()
            index = int(data_list[0])
            value = float(data_list[8])
            self.__X0[index - 1][0] = value
        pass

    def readNORMAL_EQUATION_VECTOR(self, snx):
        while True:
            data = snx.readline()
            if data.split()[0] == '-SOLUTION/NORMAL_EQUATION_VECTOR':
                break
            if data.split()[0] == '*INDEX':
                continue
            data_list = data.split()
            index = int(data_list[0])
            value = float(data_list[8])
            self.__L[index - 1][0] = value
        pass

    def readNORMAL_EQUATION_MATRIX(self, snx):
        while True:
            data = snx.readline()
            if data.split()[0] == '-SOLUTION/NORMAL_EQUATION_MATRIX':
                break
            if data.split()[0] == '*PARA1':
                continue
            data_list = data.split()
            row = int(data_list[0])
            col = int(data_list[1])
            if col == self.__unknowns:
                self.__N[row - 1][col - 1] = float(data_list[2])
            elif col == self.__unknowns - 1:
                self.__N[row - 1][col - 1] = float(data_list[2])
                self.__N[row - 1][col] = float(data_list[3])
            else:
                self.__N[row - 1][col - 1] = float(data_list[2])
                self.__N[row - 1][col] = float(data_list[3])
                self.__N[row - 1][col + 1] = float(data_list[4])
        pass

    def inputPath(self, path):
        with open(path, 'r') as snx:
            while True:
                content = snx.readline()
                if content == '%ENDSNX':
                    break
                if content[0] == '+':
                    if content.split()[0] == '+FILE/COMMENT':
                        self.readComment(snx)
                    if content.split()[0] == '+SOLUTION/STATISTICS':
                        self.readSTATISTICS(snx)
                    if content.split()[0] == '+SOLUTION/ESTIMATE':
                        self.readESTIMATE(snx)
                    if content.split()[0] == '+SOLUTION/APRIORI':
                        self.readAPRIORI(snx)
                    if content.split()[0] == '+SOLUTION/NORMAL_EQUATION_VECTOR':
                        self.readNORMAL_EQUATION_VECTOR(snx)
                    if content.split()[0] == '+SOLUTION/NORMAL_EQUATION_MATRIX':
                        self.readNORMAL_EQUATION_MATRIX(snx)
        pass

    def getVariMatrix(self, minN, maxN):
        self.maxN = maxN
        self.minN = minN
        for i in range(self.__unknowns):
            for j in range(i, self.__unknowns):
                self.__N[j][i] = self.__N[i][j]

        dx = self.__X - self.__X0

        cutoff = maxN * (maxN + 2) - 3
        S0 = (self.__weightSquare - dx.T @ self.__L) / (self.__observations - self.__unknowns)
        variMatrix = S0 * np.linalg.inv(self.__N)
        self.variMatrix = variMatrix[:cutoff, :cutoff]
        return self.variMatrix

    def getFormalError(self):
        self.formalError = np.sqrt(self.variMatrix.diagonal())
        return self.formalError

    def getIndexByDegree(self):
        Nmin = self.minN
        Nmax = self.maxN
        indexC = []
        indexS = []
        indexCS_C = []
        indexCS_S = []
        i = -1
        for l in range(Nmin, Nmax + 1):
            '''cycle for C S'''
            for m in range(0, l + 1):
                a = int((l + 1) * l / 2 + m)
                indexC.append(a)
                i += 1
                indexCS_C.append(i)

                if m != 0:
                    b = int((l + 1) * l / 2 + m)
                    indexS.append(b)
                    i += 1
                    indexCS_S.append(i)

        return np.array(indexC), np.array(indexS), np.array(indexCS_C), np.array(indexCS_S)

    def getIndexByOrder(self):
        Nmin = self.minN
        Nmax = self.maxN
        indexC = []
        indexS = []
        indexCS_C = []
        indexCS_S = []
        i = -1
        for m in range(0, Nmax + 1):
            '''cycle for C S'''
            for l in range(m, Nmax + 1):
                if l >= Nmin:
                    a = int((l + 1) * l / 2 + m)
                    indexC.append(a)
                    i += 1
                    indexCS_C.append(i)

                    if m != 0:
                        b = int((l + 1) * l / 2 + m)
                        indexS.append(b)
                        i += 1
                        indexCS_S.append(i)

        return np.array(indexC), np.array(indexS), np.array(indexCS_C), np.array(indexCS_S)

    def sortFormalErrorByDegree(self, CS):
        indexC, indexS, indexCS_C, indexCS_S = self.getIndexByDegree()
        Nmax = self.maxN
        Nmin = self.minN

        assert np.ndim(CS) == 1
        assert len(CS) == (Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1)

        CS = np.array(CS)
        size = int((Nmax + 1) * (Nmax + 2) / 2)
        C, S = np.zeros(size, dtype=np.float64), np.zeros(size, dtype=np.float64)

        C[indexC] = CS[indexCS_C]
        S[indexS] = CS[indexCS_S]

        return C, S

    def sortFormalErrorByOrder(self, CS):
        indexC, indexS, indexCS_C, indexCS_S = self.getIndexByOrder()
        Nmax = self.maxN
        Nmin = self.minN

        assert np.ndim(CS) == 1
        assert len(CS) == (Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1)

        CS = np.array(CS)
        size = int((Nmax + 1) * (Nmax + 2) / 2)
        C, S = np.zeros(size, dtype=np.float64), np.zeros(size, dtype=np.float64)

        C[indexC] = CS[indexCS_C]
        S[indexS] = CS[indexCS_S]

        return C, S

    def sortVariMatrixByDegree(self, VariMat):
        indexC, indexS, indexCS_C, indexCS_S = self.getIndexByOrder()
        Nmax = self.maxN
        Nmin = self.minN

        assert np.ndim(VariMat) == 2
        assert len(VariMat) == (Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1)

        VariMat = np.array(VariMat)

        row = int((Nmax + 1) * (Nmax + 2) / 2)
        column = (Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1)
        matrixC = np.zeros((row, row), dtype=np.float64)
        matrixS = np.zeros((row, row), dtype=np.float64)

        matrixC[indexC, :] = VariMat[indexCS_C, :]
        matrixS[indexS, :] = VariMat[indexCS_S, :]

        matrix = np.vstack((VariMat[indexCS_C, :], VariMat[indexCS_S, :]))
        for i in range(len(matrix)):
            for j in range(i, len(matrix)):
                matrix[j][i] = matrix[i][j]
        print(np.shape(matrix))
        np.save(('VariMatrix-%s.npy' % str(Nmax)), matrix)
        return matrix

    def sortVariMatrixByOrder(self, C, S, VariMat):
        C_index = []
        S_index = []
        C = MathTool.cs_1dto2d(C, MathTool.CS1dSortedBy.Degree)
        S = MathTool.cs_1dto2d(S)
        for l in range(0, len(C)):
            for m in range(l, len(C)):
                if m >= 2:
                    C_index.append(int(C[m][l]))
                    if l != 0:
                        S_index.append(int(S[m][l]))

        Nmax = 60
        Nmin = 2

        VariMat = np.array(VariMat)

        column = (Nmax + 1) * (Nmax + 2) - Nmin * (Nmin + 1) - (Nmax - Nmin + 1)
        matrix = np.zeros((column, column), dtype=np.float64)
        index = np.hstack((C_index, S_index))
        for i in range(0, len(VariMat)):
            for j in range(0, len(VariMat)):
                matrix[i, j] = VariMat[index[i], index[j]]

        np.save('SortByOrderMatrix.npy', matrix)


def load_CovMatrix(filepath, lmax: int, lmin: int = 0, get_dates=False):
    date_begin, date_end = None, None
    if get_dates:
        pat = r"(\d{4})-(\d{2})"
        re_search = re.search(pat, filepath.name)
        assert re_search is not None, f"cannot match year-month in filename {filepath.name}"

        year, month = map(int, re_search.groups())
        date_begin = date(year, month, 1)
        date_end = TimeTool.get_the_final_day_of_this_month(year=year, month=month)

    sinex = ReadSinex()
    sinex.inputPath(filepath)
    covmat = sinex.getVariMatrix(minN=lmin, maxN=lmax)
    covmat_resorted = MathTool.sort_covariance_matrix(covmat, lmax_input=lmax, lmin_input=2)

    if get_dates:
        return covmat_resorted, date_begin, date_end
    else:
        return covmat_resorted


def demo():
    from pysrc.auxiliary.aux_tool.FileTool import FileTool

    # filepath = FileTool.get_project_dir("data/L2_SH_products/VarGSM/ITSG/Grace2018/2009/ITSG-Grace2018_n96_2009-06.snx")
    filepath = pathlib.Path(
        "/Volumes/ShuhaoWork/SaGEA_Data/data/L2_SH_products/VarGSM/ITSG/Grace2018/n96/2002/ITSG-Grace2018_n96_2002-04.snx")
    covmat, date_begin, date_end = load_CovMatrix(filepath, lmax=60, get_dates=True)
    print(np.shape(covmat), date_begin, date_end)
    pass


if __name__ == '__main__':
    demo()
