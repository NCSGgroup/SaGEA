#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2025/12/29 15:26 
# @File    : gfc_reader.py
import numpy as np
from pathlib import Path

from src.sagea.utils import MathTool


def read_gfc(filepath: Path, key="gfc", lmax=None, col_indices=None):
    """
    Read a gravity field coefficient file (GFC format and its variants).

    Parameters
    ----------
    filepath : str
        Path to the gravity file.
    key : str, optional
        The leading string identifier for filtering data rows.
        Default is "gfc" (ICGEM standard).
        If None or empty string, it assumes all non-comment rows are data.
    lmax : int, optional
        Maximum degree for truncation.
        If None, the maximum degree is inferred from the file.
        (Specifying this is recommended for speed and consistent matrix size).
    col_indices : list or tuple, optional
        Column indices (0-based) representing [l, m, C, S].
        - If key is present (e.g., "gfc"), default is [2, 3, 4, 5] (skipping the key column).
        - If key is None, default is [1, 2, 3, 4].

    Returns
    -------
    cs : np.ndarray, shape ((lmax+1)**2, ), cs coefficients array, sorted as
            [c[0,0]; s[1,1], c[1,0], c[1,1]; s[2,2], s[2,1], c[2,0], c[2,1], c[2,2]; s[3,3], s[3,2], s[3,1], c[3,0], ...].
    """

    assert filepath.exists(), f"{filepath} does not exist"

    def are_all_num(x: list):
        for i in col_indices:
            if x[i - 1].replace('e', '').replace('E', '').replace('E', '').replace('E', '').replace('-',
                                                                                                    '').replace(
                '+', '').replace('.', '').isnumeric():
                pass
            else:
                return False

        return True

    if col_indices is None:
        col_indices = [1, 2, 3, 4] if key == "" else [2, 3, 4, 5]

    l_queue = col_indices[0]
    m_queue = col_indices[1]
    c_queue = col_indices[2]
    s_queue = col_indices[3]

    mat_shape = (lmax + 1, lmax + 1)
    clm, slm = np.zeros(mat_shape), np.zeros(mat_shape)

    with open(filepath) as f:
        txt_list = f.readlines()

        for i in range(len(txt_list)):
            if txt_list[i].replace(" ", "").startswith(key):
                this_line = txt_list[i].split()

                # if len(this_line) == 4 and are_all_num(this_line):
                if are_all_num(this_line):
                    l = int(this_line[l_queue - 1])
                    if l > lmax:
                        continue

                    m = int(this_line[m_queue - 1])

                    clm[l, m] = float(this_line[c_queue - 1])
                    slm[l, m] = float(this_line[s_queue - 1])

                else:
                    continue

    cs = MathTool.cs_combine_to_triangle_1d(clm, slm)
    return cs
