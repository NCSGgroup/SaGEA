# src/sagea/filtering/ddk.py

from __future__ import annotations

import warnings
from enum import Enum
from pathlib import Path
import struct
import sys

import numpy as np
from scipy.linalg import block_diag

from sagea.filtering.base import ensure_cs_3d, restore_cs_dimension
from sagea.utils.data_collecting import download_ddk_data


class DDKFilterType(Enum):
    DDK1 = 1
    DDK2 = 2
    DDK3 = 3
    DDK4 = 4
    DDK5 = 5
    DDK6 = 6
    DDK7 = 7
    DDK8 = 8


_DDK_FILENAME = {
    DDKFilterType.DDK1: "Wbd_2-120.a_1d14p_4",
    DDKFilterType.DDK2: "Wbd_2-120.a_1d13p_4",
    DDKFilterType.DDK3: "Wbd_2-120.a_1d12p_4",
    DDKFilterType.DDK4: "Wbd_2-120.a_5d11p_4",
    DDKFilterType.DDK5: "Wbd_2-120.a_1d11p_4",
    DDKFilterType.DDK6: "Wbd_2-120.a_5d10p_4",
    DDKFilterType.DDK7: "Wbd_2-120.a_1d10p_4",
    DDKFilterType.DDK8: "Wbd_2-120.a_5d9p_4",
}


def normalize_ddk_type(ddk_type: DDKFilterType | int | str) -> DDKFilterType:
    if isinstance(ddk_type, DDKFilterType):
        return ddk_type

    if isinstance(ddk_type, int):
        return DDKFilterType[f"DDK{ddk_type}"]

    if isinstance(ddk_type, str):
        name = ddk_type.upper()
        if not name.startswith("DDK"):
            name = f"DDK{name}"
        return DDKFilterType[name]

    raise TypeError(f"Unsupported DDK type: {ddk_type}")


def read_bin(file: str | Path, mode: str = "packed") -> dict:
    """
    Read DDK binary file.

    This function is adapted from the original implementation.
    """
    file = Path(file)

    if mode == "packed":
        unpack = False
    elif mode == "full":
        unpack = True
    else:
        raise ValueError("Only 'packed' or 'full' are available.")

    endian = sys.byteorder

    if endian != "little":
        raise RuntimeError(
            "The endian of the binary file is little, "
            "but the endian of OS is big."
        )

    with open(file, "rb") as f:
        dat = {}

        dat["version"] = f.read(8).decode().strip()
        dat["type"] = f.read(8).decode()
        dat["descr"] = f.read(80).decode().strip()

        for key in ["nints", "ndbls", "nval1", "nval2"]:
            dat[key] = struct.unpack("<I", f.read(4))[0]

        for key in ["pval1", "pval2"]:
            dat[key] = struct.unpack("<I", f.read(4))[0]

        dat["nvec"], dat["pval2"] = 0, 1
        dat["nread"], dat["nval2"] = 0, dat["nval1"]

        nblocks = struct.unpack("<i", f.read(4))[0]

        lists = f.read(dat["nints"] * 24).decode().split()
        for element in lists:
            dat[element] = struct.unpack("<i", f.read(4))[0]

        lists = f.read(dat["ndbls"] * 24).decode().replace(":", "").split()
        for element in lists:
            dat[element] = struct.unpack("<d", f.read(8))[0]

        lists = f.read(dat["nval1"] * 24).decode()
        dat["side1_d"] = [
            lists[i: i + 24].replace("         ", "")
            for i in range(0, len(lists), 24)
        ]

        dat["blockind"] = np.array(
            struct.unpack("<" + str(nblocks) + "i", f.read(4 * nblocks))
        )

        dat["side2_d"] = dat["side1_d"]

        npack1 = dat["pval1"] * dat["pval2"]
        dat["pack1"] = np.array(
            struct.unpack("<" + str(npack1) + "d", f.read(8 * npack1))
        )

    if not unpack:
        return dat

    sz = dat["blockind"][0]
    dat["mat1"] = dat["pack1"][: sz ** 2].reshape(sz, sz).T

    shift1 = shift2 = sz ** 2

    for i in range(1, nblocks):
        sz = dat["blockind"][i] - dat["blockind"][i - 1]
        shift2 = shift1 + sz ** 2
        dat["mat1"] = block_diag(
            dat["mat1"],
            dat["pack1"][shift1:shift2].reshape(sz, sz).T,
        )
        shift1 = shift2

    del dat["pack1"]

    return dat


def filter_sh_by_ddk_matrix(
        W: dict,
        cilm: np.ndarray,
        cilm_std: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Apply one DDK matrix to one epoch C/S coefficients.

    Parameters
    ----------
    W : dict
        DDK matrix dictionary.
    cilm : np.ndarray
        Shape: (2, lmax + 1, lmax + 1)
        cilm[0] = C, cilm[1] = S.
    """
    lmax = cilm.shape[1] - 1

    lmaxfilt = W["Lmax"]
    lminfilt = W["Lmin"]

    lmaxout = min(lmax, lmaxfilt)

    cilm_filter = np.zeros_like(cilm)

    if cilm_std is not None:
        cilm_std_filter = np.zeros_like(cilm_std)
    else:
        cilm_std_filter = None

    lastblckind, lastindex = 0, 0

    for iblk in range(W["Nblocks"]):
        degree = (iblk + 1) // 2

        if degree > lmaxout:
            break

        trig = (iblk + int(iblk > 0) + 1) % 2

        sz = W["blockind"][iblk] - lastblckind

        blockn = np.identity(lmaxfilt + 1 - degree)

        lminblk = max(lminfilt, degree)
        shift = lminblk - degree

        blockn[shift:, shift:] = W["pack1"][
            lastindex: lastindex + sz ** 2
        ].reshape(sz, sz).T

        block_sub = blockn[
            : lmaxout + 1 - degree,
            : lmaxout + 1 - degree,
        ]

        if trig:
            cilm_filter[0, degree: lmaxout + 1, degree] = block_sub @ cilm[
                0,
                degree: lmaxout + 1,
                degree,
            ]
        else:
            cilm_filter[1, degree: lmaxout + 1, degree] = block_sub @ cilm[
                1,
                degree: lmaxout + 1,
                degree,
            ]

        if cilm_std is not None:
            if trig:
                cilm_std_filter[0, degree: lmaxout + 1, degree] = np.sqrt(
                    (block_sub ** 2)
                    @ cilm_std[0, degree: lmaxout + 1, degree] ** 2
                )
            else:
                cilm_std_filter[1, degree: lmaxout + 1, degree] = np.sqrt(
                    (block_sub ** 2)
                    @ cilm_std[1, degree: lmaxout + 1, degree] ** 2
                )

        lastblckind = W["blockind"][iblk]
        lastindex = lastindex + sz ** 2

    if cilm_std is None:
        return cilm_filter

    return cilm_filter, cilm_std_filter


def get_default_ddk_data_dir() -> Path:
    """
    Default DDK data directory.

    Recommended package data location:

        src/sagea/data/ddk_data/
    """
    from importlib.resources import files

    return Path(str(files("sagea.data").joinpath("ddk_data")))


def apply_ddk_filter_cqlm(
        cqlm: np.ndarray,
        sqlm: np.ndarray,
        ddk_type: DDKFilterType | int | str = DDKFilterType.DDK3,
        # data_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply DDK filtering to C/S matrices.
    """
    cqlm3, sqlm3, single = ensure_cs_3d(cqlm, sqlm)

    ddk_type = normalize_ddk_type(ddk_type)

    # if data_dir is None:
    #     data_dir = get_default_ddk_data_dir()
    # else:
    #     data_dir = Path(data_dir)
    data_dir = get_default_ddk_data_dir()

    file = data_dir / _DDK_FILENAME[ddk_type]

    if not file.exists():
        warnings.warn("Data files for DDK filtering does not exist, trying to download it.")
        download_ddk_data(file.parent.parent / "ddk_temp", file.parent.parent)

    Wbd = read_bin(file)

    c_list = []
    s_list = []

    for i in range(cqlm3.shape[0]):
        cilm = np.array([cqlm3[i], sqlm3[i]])
        cilm_filtered = filter_sh_by_ddk_matrix(Wbd, cilm)

        c_list.append(cilm_filtered[0])
        s_list.append(cilm_filtered[1])

    cqlm_f = np.asarray(c_list)
    sqlm_f = np.asarray(s_list)

    return restore_cs_dimension(cqlm_f, sqlm_f, single)


# !/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/6/17 13:15 
# @File    : ddk.py.py

if __name__ == "__main__":
    pass
