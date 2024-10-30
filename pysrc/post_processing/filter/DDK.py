"""
This file refers to ggtools package written by lcx366 2016.
For more information, please refer to https://github.com/lcx366/GGTOOLS
"""

import struct
import sys
import numpy as np
from scipy.linalg import block_diag

from enum import Enum

from pysrc.auxiliary.aux_tool.FileTool import FileTool
from pysrc.auxiliary.aux_tool.MathTool import MathTool
from pysrc.post_processing.filter.Base import SHCFilter


class DDKFilterType(Enum):
    DDK1 = 1
    DDK2 = 2
    DDK3 = 3
    DDK4 = 4
    DDK5 = 5
    DDK6 = 6
    DDK7 = 7
    DDK8 = 8


def read_BIN(file, mode='packed'):
    if mode == 'packed':
        unpack = False
    elif mode == 'full':
        unpack = True
    else:
        raise Exception("Only 'packed' or 'full' are avaliable.")

    endian = sys.byteorder

    if endian == 'little':
        f = open(file, 'rb')
    else:
        raise Exception('The endian of the binary file is little, but the endian of OS is big.')

    dat = {}
    dat['version'] = f.read(8).decode().strip()
    dat['type'] = f.read(8).decode()
    dat['descr'] = f.read(80).decode().strip()

    for key in ['nints', 'ndbls', 'nval1', 'nval2']:
        dat[key] = struct.unpack('<I', f.read(4))[0]

    for key in ['pval1', 'pval2']:
        dat[key] = struct.unpack('<I', f.read(4))[0]

    dat['nvec'], dat['pval2'] = 0, 1
    dat['nread'], dat['nval2'] = 0, dat['nval1']

    nblocks = struct.unpack('<i', f.read(4))[0]

    lists = f.read(dat['nints'] * 24).decode().split()
    for element in lists:
        dat[element] = struct.unpack('<i', f.read(4))[0]

    lists = f.read(dat['ndbls'] * 24).decode().replace(':', '').split()
    for element in lists:
        dat[element] = struct.unpack('<d', f.read(8))[0]

    lists = f.read(dat['nval1'] * 24).decode()
    dat['side1_d'] = [(lists[i:i + 24]).replace('         ', '') for i in range(0, len(lists), 24)]

    dat['blockind'] = np.array(struct.unpack('<' + str(nblocks) + 'i', f.read(4 * nblocks)))

    dat['side2_d'] = dat['side1_d']

    npack1 = dat['pval1'] * dat['pval2']
    dat['pack1'] = np.array(struct.unpack('<' + str(npack1) + 'd', f.read(8 * npack1)))

    f.close()

    if not unpack: return dat

    sz = dat['blockind'][0]
    dat['mat1'] = dat['pack1'][:sz ** 2].reshape(sz, sz).T

    shift1 = shift2 = sz ** 2

    for i in range(1, nblocks):
        sz = dat['blockind'][i] - dat['blockind'][i - 1]
        shift2 = shift1 + sz ** 2
        dat['mat1'] = block_diag(dat['mat1'], dat['pack1'][shift1:shift2].reshape(sz, sz).T)
        shift1 = shift2
    del dat['pack1']

    return dat


def filterSH(W, cilm, cilm_std=None):
    lmax = cilm.shape[1] - 1

    lmaxfilt, lminfilt = W['Lmax'], W['Lmin']

    lmaxout = min(lmax, lmaxfilt)

    cilm_filter = np.zeros_like(cilm)
    cilm_std_filter = np.zeros_like(cilm_std)

    lastblckind, lastindex = 0, 0

    for iblk in range(W['Nblocks']):
        degree = (iblk + 1) // 2

        if degree > lmaxout: break
        trig = (iblk + int(iblk > 0) + 1) % 2

        sz = W['blockind'][iblk] - lastblckind

        blockn = np.identity(lmaxfilt + 1 - degree)

        lminblk = max(lminfilt, degree)

        shift = lminblk - degree
        blockn[shift:, shift:] = W['pack1'][lastindex:lastindex + sz ** 2].reshape(sz, sz).T

        if trig:
            cilm_filter[0, degree:lmaxout + 1, degree] = np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree],
                                                                cilm[0, degree:lmaxout + 1, degree])
        else:
            cilm_filter[1, degree:lmaxout + 1, degree] = np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree],
                                                                cilm[1, degree:lmaxout + 1, degree])

        if cilm_std is not None:
            if trig:
                cilm_std_filter[0, degree:lmaxout + 1, degree] = np.sqrt(
                    np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree] ** 2,
                           cilm_std[0, degree:lmaxout + 1, degree] ** 2))
            else:
                cilm_std_filter[1, degree:lmaxout + 1, degree] = np.sqrt(
                    np.dot(blockn[:lmaxout + 1 - degree, :lmaxout + 1 - degree] ** 2,
                           cilm_std[1, degree:lmaxout + 1, degree] ** 2))

        lastblckind = W['blockind'][iblk]
        lastindex = lastindex + sz ** 2
        pass

    if cilm_std is None:
        return cilm_filter
    else:
        return cilm_filter, cilm_std_filter


class DDKConfig:
    def __init__(self):
        self.filter_type = DDKFilterType.DDK3

    def set_filter_type(self, filter_type: DDKFilterType):

        if type(filter_type) is DDKFilterType:
            self.filter_type = filter_type

        elif type(filter_type) is int:
            ddk_type = DDKFilterType.__members__[f'DDK{str(filter_type)}']
            self.filter_type = ddk_type

        else:
            return -1

        return self


class DDK(SHCFilter):
    def __init__(self):
        self.configuration = DDKConfig()

    def __apply_to_csqlm(self, cqlm, sqlm):
        ddktype = self.configuration.filter_type

        ddk_data_dir = FileTool.get_project_dir() / 'data/ddk_data'
        if ddktype == DDKFilterType.DDK1:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_1d14p_4')
        elif ddktype == DDKFilterType.DDK2:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_1d13p_4')
        elif ddktype == DDKFilterType.DDK3:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_1d12p_4')
        elif ddktype == DDKFilterType.DDK4:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_5d11p_4')
        elif ddktype == DDKFilterType.DDK5:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_1d11p_4')
        elif ddktype == DDKFilterType.DDK6:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_5d10p_4')
        elif ddktype == DDKFilterType.DDK7:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_1d10p_4')
        elif ddktype == DDKFilterType.DDK8:
            Wbd = read_BIN(ddk_data_dir / 'Wbd_2-120.a_5d9p_4')
        else:
            raise Exception

        assert len(cqlm) == len(sqlm)
        cilms_filtered = []
        for i in range(len(cqlm)):
            cilm = np.array([cqlm[i], sqlm[i]])
            cilms_filtered.append(filterSH(Wbd, cilm))  # list, [ [c1,s1], [c2,s2], ... ]

        cqlm = np.array([cilms_filtered[i][0] for i in range(len(cilms_filtered))])
        sqlm = np.array([cilms_filtered[i][1] for i in range(len(cilms_filtered))])
        return cqlm, sqlm

    def apply_to(self, cqlm, sqlm):
        cqlm, sqlm, single = self._cs_to_3d_array(cqlm, sqlm)

        cqlm_f, sqlm_f = self.__apply_to_csqlm(cqlm, sqlm)

        if single:
            assert cqlm_f.shape[0] == sqlm_f.shape[0] == 1
            return cqlm_f[0], sqlm_f[0]
        else:
            return cqlm_f, sqlm_f
