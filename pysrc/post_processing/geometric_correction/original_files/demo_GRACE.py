"""
@Company: CGE-HUST, Wuhan, China
@Author: Yang Fan
@Contact: yfan_cge@hust.edu.cn
@Modify Time:2021/11/23
@Description:
"""
import sys

from pysrc.auxiliary.aux_tool.FileTool import FileTool

sys.path.append('../../')

from pysrc.post_processing.geometric_correction.original_files.LoadSH import LoadGsmByYear
from pysrc.post_processing.geometric_correction.original_files.LowDeg import LowDegreeReplace
from pysrc.post_processing.geometric_correction.original_files.Filtering import Gaussion
from pysrc.post_processing.geometric_correction.original_files.GC import GeometricalCorrection
from pysrc.post_processing.geometric_correction.original_files.Setting import FieldType, Assumption
import numpy as np


def demo_GRACE_OneMonth():
    """
    This is a demo for GRACE ellipsoid/topography correction with our iterative scaling method.
    A demo for only one month's data processing.
    :return:
    """

    '''define the max degree of gravity solution'''
    lmax = 90

    '''Load the monthly gravity fields'''
    # ts = LoadGsmByYear(localDir='../data/L2_SH_products/CSR', begin='2002', end='2015', opt='90')
    ts = LoadGsmByYear(localDir=FileTool.get_project_dir('data/L2_SH_products/GSM/CSR'), begin='2002', end='2015',
                       opt='90')

    '''Preparation for the replacement of degree 1 and degree 2'''
    ld = LowDegreeReplace().load(localDir=FileTool.get_project_dir('data/L2_low_degrees'))

    '''get the time-mean and replace degree 1/2'''
    C_mean = np.zeros(int((lmax + 2) * (lmax + 1) / 2))
    S_mean = np.zeros(int((lmax + 2) * (lmax + 1) / 2))
    for x in ts:
        '''low degree replacement'''
        ld.setGrav(x).rmDegZero().rpDegOne().rpDegTwo().rpDegThree()
        '''get the time-mean SHs'''
        C, S = x.getCS(lmax)
        C_mean += C
        S_mean += S
    C_mean = C_mean / len(ts)
    S_mean = S_mean / len(ts)

    '''remove the mean from monthly gravity fields'''
    SHC = [x.getCS(lmax)[0] - C_mean for x in ts]
    SHS = [x.getCS(lmax)[1] - S_mean for x in ts]

    '''specify one month's data as the input: the 35th monthly gravity field'''
    '''it could be an arbitrary one other than '35'.'''
    GivenMonth = 35
    print('\nStart the ellipsoid and topography correction for Month: %s to %s'
          % (ts[GivenMonth].date_begin, ts[GivenMonth].date_end))
    input = [SHC[GivenMonth], SHS[GivenMonth]]

    '''Gauss filter'''
    Gs = Gaussion().setRadius(300, lmax)
    input[0], input[1] = Gs.setCS(input[0], input[1]).getCS()

    '''define the griding type'''
    lat = np.arange(90, -90.1, -0.5)
    lon = np.arange(0, 360, 0.5)

    '''using the iterative scaling method to undertake the ellipsoid/topography correction'''
    gc = GeometricalCorrection().configure(Nmax=lmax, lat=lat, lon=lon,
                                           assumption=Assumption.ActualEarth, kind=FieldType.EWH)

    '''obtain the corrected gravity fields in terms of spherical harmonic coefficients'''
    output = gc.setInput(GravityField=input).correct()

    '''Optionally, John Wahr's formulation can be applied to the output above to derive the correct surface mass'''
    '''load the pre-computed result, to check if the code works well'''
    validation = np.load('Output_verified.npy')

    '''make a validation'''
    # if np.max(output - validation) == 0.0:
    if np.max(output - validation) < 1e-20:
        print('\nThe code is correctly configured')

    pass


if __name__ == '__main__':
    demo_GRACE_OneMonth()
