import copy

import numpy as np

from pysrc.auxiliary.aux_tool.MathTool import MathTool


class CoreSHC:
    """
    This class is to store the spherical harmonic coefficients for the use in necessary data processing.

    Attribute self.cs stores the coefficients in 1d-array combined with c and s, or 2d-array for series.
    which are sorted by degree, for example,
    [c[0,0]; s[1,1], c[1,0], c[1,1]; s[2,2], s[2,1], c[2,0], c[2,1], c[2,2]; s[3,3], s[3,2], s[3,1], c[3,0], ...],
    or
    [[c1[0,0]; s1[1,1], c1[1,0], c1[1,1]; s1[2,2], s1[2,1], c1[2,0], c1[2,1], c1[2,2]; ...],
     [c2[0,0]; s2[1,1], c2[1,0], c2[1,1]; s2[2,2], s2[2,1], c2[2,0], c2[2,1], c2[2,2]; ...],
     [                                        ...                                         ]]
    """

    def __init__(self, c, s=None):
        """

        :param c: harmonic coefficients c in 2-dimension (l,m), or a series (q,l,m),
                or 1-dimension array sorted by degree [c00, s11, c10, c11, s22, s21, ...] if s is None.
        :param s: harmonic coefficients s in 2-dimension (l,m), or a series (q,l,m),
                or None.
        """
        if s is None:
            self.value = np.array(c)
        else:
            assert np.shape(c) == np.shape(s)

            if len(np.shape(c)) == 2:
                self.value = MathTool.cs_combine_to_triangle_1d(c, s)

            elif len(np.shape(c)) == 3:
                cs = []
                for i in range(np.shape(c)[0]):
                    this_cs = MathTool.cs_combine_to_triangle_1d(c[i], s[i])
                    cs.append(this_cs)
                self.value = np.array(cs)

        if len(np.shape(self.value)) == 1:
            self.value = self.value[None, :]

        assert len(np.shape(self.value)) == 2

    def append(self, *params):
        """

        :param params: One parameter of instantiated SHC,
         or two parameters of c and s with the same input requirement as SHC.
        :return:
        """
        assert len(params) in (1, 2)

        if len(params) == 1:
            if issubclass(type(params[0]), CoreSHC):
                shc = params[0]
            else:
                shc = CoreSHC(params[0])

        else:
            shc = CoreSHC(*params)

        assert np.shape(shc.value)[-1] == np.shape(self.value)[-1]

        self.value = np.concatenate([self.value, shc.value])
        return self

    def is_series(self):
        """
        To determine whether the spherical harmonic coefficients stored in this class are one group or multiple groups.
        :return: bool, True if it stores multiple groups, False if it stores only one group.
        """
        return np.shape(self.value)[0] != 1

    def get_lmax(self):
        """

        :return: int, max degree/order of the spherical harmonic coefficients stored in this class.
        """
        length_of_cs1d = np.shape(self.value)[-1]
        lmax = int(np.sqrt(length_of_cs1d) - 1)

        return lmax

    def get_cs2d(self):
        """
        return: cqlm, sqlm. Both cqlm and sqlm are 3-dimension, EVEN IF NOT self.is_series()
        """
        lmax = self.get_lmax()

        num_of_series = np.shape(self.value)[0]
        cqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))
        sqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))

        for i in range(num_of_series):
            this_cs = self.value[i]
            this_clm, this_slm = MathTool.cs_decompose_triangle1d_to_cs2d(this_cs)
            cqlm[i, :, :] = this_clm
            sqlm[i, :, :] = this_slm

        return cqlm, sqlm

    def __de_average(self):
        if self.is_series():
            self.value -= np.mean(self.value, axis=0)
        else:
            raise Exception

    def de_background(self, background=None):
        """
        if background is None, de average
        """
        if background is None:
            self.__de_average()

        else:
            assert issubclass(type(background), CoreSHC)
            self.value -= background.value

    @staticmethod
    def identity(lmax: int):
        basis_num = (lmax + 1) ** 2
        cs = np.eye(basis_num)

        return CoreSHC(cs)

    def __add__(self, other):
        assert issubclass(type(other), CoreSHC)

        return CoreSHC(self.value + other.value)

    def __sub__(self, other):
        assert issubclass(type(other), CoreSHC)

        return CoreSHC(self.value - other.value)

    def add(self, shc, lbegin=None, lend=None):
        if (lend is None) and (lbegin is None):
            self.value += shc.value
            return self

        else:
            assert (lend is None) or (0 <= lend <= self.get_lmax())
            assert (lbegin is None) or (0 <= lbegin <= self.get_lmax())

            shc_copy = copy.deepcopy(shc)

            if lend is not None:
                shc_copy.value[:, (lend + 1) ** 2:] = 0
            if lbegin is not None:
                shc_copy.value[:, :lbegin ** 2] = 0

            return self.add(shc_copy)

    def subtract(self, shc, lbegin=None, lend=None):
        if (lend is None) and (lbegin is None):
            self.value -= shc.value
            return self

        else:
            assert (lend is None) or (0 <= lend <= self.get_lmax())
            assert (lbegin is None) or (0 <= lbegin <= self.get_lmax())

            shc_copy = copy.deepcopy(shc)

            if lend is not None:
                shc_copy.value[:, (lend + 1) ** 2:] = 0
            if lbegin is not None:
                shc_copy.value[:, :lbegin ** 2] = 0

            return self.subtract(shc_copy)
