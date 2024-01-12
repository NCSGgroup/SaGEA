import numpy as np

from pysrc.auxiliary.tools.MathTool import MathTool


class SHC:
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
                or cs1d sorted by degree [c00, s11, c10, c11, s22, s21, ...] if s is None.
        :param s: harmonic coefficients s in 2-dimension (l,m), or a series (q,l,m),
                or None.
        """
        if s is None:
            self.cs = c
        else:
            assert np.shape(c) == np.shape(s)

            if len(np.shape(c)) == 2:
                self.cs = MathTool.cs_combine_to_triangle_1d(c, s)

            elif len(np.shape(c)) == 3:
                cs = []
                for i in range(np.shape(c)[0]):
                    this_cs = MathTool.cs_combine_to_triangle_1d(c[i], s[i])
                    cs.append(this_cs)
                self.cs = np.array(cs)

    def append(self, *params):
        """

        :param params: One parameter of instantiated SHC or two parameters of c and s in 2-dimension
        :return:
        """
        assert len(params) in (1, 2)

        if len(params) == 1:
            assert type(params[0]) is SHC

            shc = params[0]

        else:
            c2d, s2d = params[0], params[1]
            assert len(np.shape(c2d)) == len(np.shape(s2d)) == 2

            shc = SHC(c2d, s2d)

        assert np.shape(shc.cs)[-1] == np.shape(self.cs)[-1]

        array_to_append = shc.cs if shc.is_series() else np.array([shc.cs])
        array_self = self.cs if self.is_series() else [self.cs]

        self.cs = np.concatenate([array_self, array_to_append])
        return self

    def is_series(self):
        """
        To determine whether the spherical harmonic coefficients stored in this class are one group or multiple groups.
        :return: bool, True if it stores multiple groups, False if it stores only one group.
        """
        return len(np.shape(self.cs)) == 2

    def get_lmax(self):
        """

        :return: int, max degree/order of the spherical harmonic coefficients stored in this class.
        """
        length_of_cs1d = np.shape(self.cs)[-1]
        lmax = int(np.sqrt(length_of_cs1d) - 1)

        return lmax

    def get_cs2d(self):
        """
        return: cqlm, sqlm. both cqlm and sqlm are 3-dimension even if not self.is_series()
        """
        lmax = self.get_lmax()

        if self.is_series():
            num_of_series = np.shape(self.cs)[0]
            cqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))
            sqlm = np.zeros((num_of_series, lmax + 1, lmax + 1))

            for i in range(num_of_series):
                this_cs = self.cs[i]
                this_clm, this_slm = MathTool.cs_decompose_triangle1d_to_cs2d(this_cs)
                cqlm[i, :, :] = this_clm
                sqlm[i, :, :] = this_slm

            return cqlm, sqlm

        else:
            clm, slm = MathTool.cs_decompose_triangle1d_to_cs2d(self.cs)
            return np.array([clm]), np.array([slm])

    def __de_average(self):
        if self.is_series():
            self.cs -= np.mean(self.cs, axis=0)
        else:
            raise Exception

    def de_background(self, background=None):
        """
        if background is None, de average
        """
        if background is None:
            self.__de_average()

        else:
            assert type(background) is SHC
            self.cs -= background.cs

    @staticmethod
    def eye(lmax: int):
        basis_num = (lmax + 1) ** 2
        cs = np.eye(basis_num)

        return SHC(cs)