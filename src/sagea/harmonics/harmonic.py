#!/use/bin/env python
# coding=utf-8
# @Author  : Shuhao Liu
# @Time    : 2026/6/17 13:56 
# @File    : harmonic.py

from __future__ import annotations

from enum import Enum

import numpy as np

from sagea.constants.constant import PhysicalDimension
from sagea.utils.MathTool import MathTool
from sagea.physics.quadratures import GLQ, DH2, DH


class GRDType(Enum):
    DH = 1
    DH2 = 2
    GLQ = 3


class Harmonic:
    """
    Spherical harmonic analysis/synthesis on fixed grids.

    Supported grids:
    - GLQ
    - DH
    - DH2
    - custom regular lat/lon grid

    Notes
    -----
    lat/lon input should be in degrees.
    Internal colat/lon are stored in radians.
    """

    def __init__(
            self,
            lmax: int,
            grid_type: GRDType | None = GRDType.GLQ,
            lat: np.ndarray | None = None,
            lon: np.ndarray | None = None,
    ):
        self.lmax = int(lmax)

        if self.lmax < 0:
            raise ValueError("lmax must be non-negative.")

        if (lat is None) != (lon is None):
            raise ValueError("lat and lon should be both None or both provided.")

        if lat is not None and grid_type is not None:
            raise ValueError("For custom lat/lon grid, grid_type must be None.")

        if lat is None:
            if grid_type is None:
                raise ValueError("grid_type cannot be None when lat/lon are not provided.")

            self.__grid_type = grid_type
            self.colat, self.lon, self.analysis_weight = self.__get_grid_nodes(
                self.lmax, grid_type
            )

        else:
            self.__grid_type = None

            self.colat, self.lon = MathTool.get_colat_lon_rad(lat, lon)

            # DH-like latitude weights for custom regular grids.
            nlat = len(self.colat)
            wi_dh = np.ones_like(self.colat, dtype=float)

            for j, theta in enumerate(self.colat):
                wi_dh[j] = np.sum(
                    [
                        np.sin((2 * ll + 1) * theta) / (2 * ll + 1)
                        for ll in range(int(nlat / 2 - 1))
                    ]
                )

            wi_dh *= 4 / np.pi
            self.analysis_weight = np.sin(self.colat) * wi_dh

        self.analysis_weight = np.asarray(self.analysis_weight, dtype=float)
        self.analysis_weight /= np.sum(self.analysis_weight)

        self.__prepare()

    @property
    def grid_type(self) -> GRDType | None:
        return self.__grid_type

    @staticmethod
    def __get_grid_nodes(lmax: int, grid_type: GRDType):
        if grid_type == GRDType.GLQ:
            return GLQ.get_nodes(lmax)

        if grid_type == GRDType.DH2:
            return DH2.get_nodes(lmax)

        if grid_type == GRDType.DH:
            return DH.get_nodes(lmax)

        raise ValueError(f"Unsupported grid_type: {grid_type}")

    def __prepare(self):
        self.nlat = len(self.colat)
        self.nlon = len(self.lon)

        self.pilm = MathTool.get_Legendre(self.colat, self.lmax)
        # shape: (nlat, lmax + 1, lmax + 1)

        m = np.arange(self.lmax + 1)
        self.g = m[:, None] @ self.lon[None, :]
        # shape: (lmax + 1, nlon)

    def synthesis(
            self,
            cqlm: np.ndarray,
            sqlm: np.ndarray,
            special_type: PhysicalDimension | None = None,
    ) -> np.ndarray:
        """
        SHC -> evaluation at discrete points.

        Parameters
        ----------
        cqlm, sqlm : ndarray
            Shape can be:
            - (lmax + 1, lmax + 1)
            - (time, lmax + 1, lmax + 1)

        Returns
        -------
        ndarray
            Shape:
            - (nlat, nlon) for single input
            - (time, nlat, nlon) for series input
        """

        if cqlm.shape != sqlm.shape:
            raise ValueError("cqlm and sqlm should have the same shape.")

        if cqlm.ndim not in (2, 3):
            raise ValueError("cqlm/sqlm should be 2D or 3D arrays.")

        if special_type is not None:
            raise NotImplementedError("special_type synthesis is not implemented yet.")

        single = cqlm.ndim == 2

        if single:
            cqlm = cqlm[None, :, :]
            sqlm = sqlm[None, :, :]

        cqlm = np.asarray(cqlm, dtype=float)
        sqlm = np.asarray(sqlm, dtype=float)

        am = np.einsum("tlm,ilm->tim", cqlm, self.pilm)
        bm = np.einsum("tlm,ilm->tim", sqlm, self.pilm)

        co = np.cos(self.g)
        so = np.sin(self.g)

        grid = np.einsum("tim,mj->tij", am, co) + np.einsum("tim,mj->tij", bm, so)

        return grid[0] if single else grid

    def analysis(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Grid -> SHC.

        Parameters
        ----------
        grid : ndarray
            Shape:
            - (nlat, nlon)
            - (time, nlat, nlon)

        Returns
        -------
        cqlm, sqlm : tuple[ndarray, ndarray]
            Shape:
            - (lmax + 1, lmax + 1) for single input
            - (time, lmax + 1, lmax + 1) for series input
        """

        if grid.ndim not in (2, 3):
            raise ValueError("grid should be a 2D or 3D array.")

        single = grid.ndim == 2

        if single:
            grid = grid[None, :, :]

        grid = np.asarray(grid, dtype=float)

        if grid.shape[-2:] != (self.nlat, self.nlon):
            raise ValueError(
                f"grid shape mismatch. Expected (..., {self.nlat}, {self.nlon}), "
                f"got {grid.shape}."
            )

        co = np.cos(self.g)
        so = np.sin(self.g)

        factor_lon = 1 / self.nlon

        am = np.einsum("tij,mj->tim", grid, co) * factor_lon
        bm = np.einsum("tij,mj->tim", grid, so) * factor_lon

        w = self.analysis_weight

        cqlm = np.einsum("tim,ilm,i->tlm", am, self.pilm, w)
        sqlm = np.einsum("tim,ilm,i->tlm", bm, self.pilm, w)

        return (cqlm[0], sqlm[0]) if single else (cqlm, sqlm)

    def analysis_degree(
            self,
            gqij: np.ndarray,
            degree: int,
    ):
        """
        Analyze only one spherical harmonic degree.

        Parameters
        ----------
        gqij : ndarray
            Shape:
                - (nlat, nlon)
                - (ntime, nlat, nlon)

        degree : int
            Target degree l.

        Returns
        -------
        clm, slm : ndarray
            Shape:
                - (degree + 1,) if input single grid
                - (ntime, degree + 1) if input multiple grids

            They correspond to m = 0, 1, ..., degree.
        """

        if not (0 <= degree <= self.lmax):
            raise ValueError(
                f"degree should be in [0, {self.lmax}], got {degree}."
            )

        gqij = np.asarray(gqij, dtype=float)

        single = gqij.ndim == 2
        if single:
            gqij = gqij[None, :, :]

        if gqij.ndim != 3:
            raise ValueError(
                f"gqij should be 2D or 3D, got shape {gqij.shape}."
            )

        if gqij.shape[1:] != (self.nlat, self.nlon):
            raise ValueError(
                f"Grid shape mismatch. Expected {(self.nlat, self.nlon)}, "
                f"got {gqij.shape[1:]}."
            )

        m = np.arange(degree + 1)

        co = np.cos(self.g[m])
        so = np.sin(self.g[m])

        factor_lon = 1.0 / self.nlon

        am = np.einsum(
            "qij,mj->qim",
            gqij,
            co,
            optimize="greedy",
        ) * factor_lon

        bm = np.einsum(
            "qij,mj->qim",
            gqij,
            so,
            optimize="greedy",
        ) * factor_lon

        pilm_l = self.pilm[:, degree, :degree + 1]
        weight = self.analysis_weight

        clm = np.einsum(
            "qim,im,i->qm",
            am,
            pilm_l,
            weight,
            optimize="greedy",
        )

        slm = np.einsum(
            "qim,im,i->qm",
            bm,
            pilm_l,
            weight,
            optimize="greedy",
        )

        if single:
            return clm[0], slm[0]

        return clm, slm


class HarmonicDiscrete:
    """
    Spherical harmonic synthesis on discrete points.
    Notes
    -----
    lat/lon input should be in degrees and in same length.
    Internal colat/lon are stored in radians.
    """

    def __init__(
            self,
            lmax: int,
            lat: np.ndarray,
            lon: np.ndarray,
    ):
        self.lmax = int(lmax)

        if self.lmax < 0:
            raise ValueError("lmax must be non-negative.")

        else:
            self.colat, self.lon = MathTool.get_colat_lon_rad(lat, lon)

        self.__prepare()

    def __prepare(self):
        self.nlat = len(self.colat)
        self.nlon = len(self.lon)

        self.pilm = MathTool.get_Legendre(self.colat, self.lmax)
        # shape: (nlat, lmax + 1, lmax + 1)

        m = np.arange(self.lmax + 1)
        self.g = m[:, None] @ self.lon[None, :]
        # shape: (lmax + 1, nlon)

    def synthesis(
            self,
            cqlm: np.ndarray,
            sqlm: np.ndarray,
    ) -> np.ndarray:
        """
        SHC -> evaluation at discrete points.

        Parameters
        ----------
        cqlm, sqlm : ndarray
            Shape can be:
            - (lmax + 1, lmax + 1)
            - (time, lmax + 1, lmax + 1)

        Returns
        -------
        ndarray
            Shape:
            - (npoints,) for single input
            - (time, npoints,) for series input
        """

        if cqlm.shape != sqlm.shape:
            raise ValueError("cqlm and sqlm should have the same shape.")

        if cqlm.ndim not in (2, 3):
            raise ValueError("cqlm/sqlm should be 2D or 3D arrays.")

        single = cqlm.ndim == 2

        if single:
            cqlm = cqlm[None, :, :]
            sqlm = sqlm[None, :, :]

        cqlm = np.asarray(cqlm, dtype=float)
        sqlm = np.asarray(sqlm, dtype=float)

        am = np.einsum("tlm,ilm->tim", cqlm, self.pilm)
        bm = np.einsum("tlm,ilm->tim", sqlm, self.pilm)

        co = np.cos(self.g)
        so = np.sin(self.g)

        value_at_points = np.einsum("qil,li->qi", am, co) + np.einsum("qil,li->qi", bm, so)

        return value_at_points[0] if single else value_at_points
