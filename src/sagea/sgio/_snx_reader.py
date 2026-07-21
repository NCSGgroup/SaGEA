#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/7/18 10:58 
# @File    : _snx_reader.py

import pathlib

import h5py
import numpy as np
import re


def parse_estimate_block(lines):
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith('+SOLUTION/ESTIMATE'):
            start = i + 1
        elif line.startswith('-SOLUTION/ESTIMATE'):
            end = i
            break

    if start is None or end is None:
        raise ValueError("Cannot find +SOLUTION/ESTIMATE block")

    params = []
    for line in lines[start:end]:
        s = line.strip()
        if not s or s.startswith('*'):
            continue
        parts = s.split()
        if len(parts) < 10:
            continue
        try:
            ptype = parts[1]
            degree = int(parts[2])
            order = int(parts[4])
            params.append((ptype, degree, order))
        except:
            continue

    return params


def build_source_index_map(params):
    """
      ('C', n, m) -> idx
      ('S', n, m) -> idx
    """
    mp = {}
    for idx, (ptype, degree, order) in enumerate(params):
        if ptype == 'CN':
            mp[('C', degree, order)] = idx
        elif ptype == 'SN':
            mp[('S', degree, order)] = idx
    return mp


def parse_matrix_block(lines, block_name):
    start = None
    end = None
    header_line = None

    for i, line in enumerate(lines):
        if line.startswith('+' + block_name):
            start = i + 1
            header_line = line.strip()
        elif line.startswith('-' + block_name):
            end = i
            break

    if start is None or end is None:
        raise ValueError(f"Cannot find +{block_name} block")

    block = lines[start:end]

    tri = 'U' if header_line and header_line.endswith('U') else 'L'

    entries = []
    max_idx = 0

    for line in block:
        s = line.strip()
        if not s or s.startswith('*'):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            i = int(parts[0]) - 1
            j = int(parts[1]) - 1
            vals = [float(x) for x in parts[2:]]
            for k, v in enumerate(vals):
                if tri == 'U':
                    row = i
                    col = j + k
                else:
                    row = i
                    col = j + k
                entries.append((row, col, v))
                max_idx = max(max_idx, row, col)
        except:
            continue

    n = max_idx + 1
    mat = np.zeros((n, n), dtype=float)
    for i, j, v in entries:
        if 0 <= i < n and 0 <= j < n:
            mat[i, j] = v
            mat[j, i] = v
    return mat


def target_order(lmax):
    order = [('C', 0, 0)]
    for n in range(1, lmax + 1):
        # Snn ... S n1
        for m in range(n, 0, -1):
            order.append(('S', n, m))
        # Cn0 ... Cnn
        for m in range(0, n + 1):
            order.append(('C', n, m))
    return order


def read_sinex_cov(sinex_path, lmax):
    with open(sinex_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    params = parse_estimate_block(lines)
    src_map = build_source_index_map(params)

    mat = parse_matrix_block(lines, 'SOLUTION/NORMAL_EQUATION_MATRIX')

    cov_src = np.linalg.inv(mat)

    tgt = target_order(lmax)
    n_tgt = (lmax + 1) ** 2
    cov_out = np.zeros((n_tgt, n_tgt), dtype=float)

    for a, pa in enumerate(tgt):
        ia = src_map.get(pa, None)
        if ia is None:
            continue
        for b, pb in enumerate(tgt):
            ib = src_map.get(pb, None)
            if ib is None:
                continue
            cov_out[a, b] = cov_src[ia, ib]

    return cov_out, tgt


if __name__ == "__main__":
    pass
