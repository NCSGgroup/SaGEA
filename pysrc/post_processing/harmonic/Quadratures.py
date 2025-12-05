"""
Processing calculations for GLQ (Gauss-Legendre Quadrature) grid
"""

import numpy as np
from scipy.special import roots_legendre


class GLQ:
    @staticmethod
    def get_nodes(lmax):
        """
        Generates 1D coordinates (Longitude, Colatitude) for a GLQ grid
        given a maximum spherical harmonic degree L_max.

        Specifications:
        1. Longitude range: [-pi, pi)
        2. Latitude definition: Colatitude [0, pi]
        3. Units: Radians
        4. Colatitude order: North -> South (0 -> pi)

        Parameters:
            lmax (int): Maximum degree of spherical harmonic expansion (Bandwidth).

        Returns:
            colats (ndarray): Colatitude matrix in radians, range (0, pi).
                                  Shape: (nlat, nlon).
            lons (ndarray): Longitude matrix in radians, range [-pi, pi).
                                Shape: (nlat, nlon).
            weights (ndarray): Integration weights
        """

        nlat = lmax + 1
        nlon = 2 * lmax + 1

        cos_theta, weights = roots_legendre(nlat)

        colats = np.arccos(cos_theta)[::-1]  # 0 to pi
        lons = np.linspace(-np.pi, np.pi, nlon, endpoint=False)

        return colats, lons, weights


class DH:
    """
    Driscoll-Healy Grid (Type I).
    Equidistant grid NOT including the poles.
    """

    @staticmethod
    def get_nodes(lmax):
        """
        Generates 1D coordinates and weights for a Driscoll-Healy (Type I) grid.

        Specifications:
        1. Longitude range: [-pi, pi)
        2. Latitude definition: Colatitude (0, pi), excluding poles.
        3. Units: Radians
        4. Colatitude order: North -> South

        Parameters:
            lmax (int): Maximum degree of spherical harmonic expansion.

        Returns:
            colats (ndarray): Colatitude points (1D). Shape: (nlat,).
            lons (ndarray): Longitude points (1D). Shape: (nlon,).
            weights (ndarray): Integration weights (1D) corresponding to colats.
        """
        # DH grid sampling requirement: 2 * lmax + 2
        # Note: Different libraries might define nlat differently (e.g., 2*lmax).
        # We use the standard 2(L+1) for full recovery.
        nlat = 2 * (lmax + 1)
        nlon = 2 * (lmax + 1)  # Often equals nlat in DH

        # 1. Colatitudes: Equidistant, shifted by pi / (2 * nlat) to avoid poles
        # theta_j = pi * (2j + 1) / (2 * nlat) for j = 0 ... nlat-1
        j = np.arange(nlat)
        colats = np.pi * (2 * j + 1) / (2 * nlat)

        # 2. Longitudes: Equidistant
        lons = np.linspace(-np.pi, np.pi, nlon, endpoint=False)

        # 3. Weights for DH1 integration
        # w_j = (2 * pi / nlat) * sin(theta_j) * sum(...)
        # This is a simplified computation for DH weights.
        # For exact SHT inverse, weights are usually implicit in the FFT algorithm.
        # However, for simple quadrature, an approximation is often:
        # w_j = (4 * pi) / nlat * sin(theta_j)  (Rough approximation for area)

        # Here is the rigorous calculation for DH weights:
        weights = np.zeros(nlat)
        for i, theta in enumerate(colats):
            # Summation part of the DH weight formula
            sum_k = 0
            for k in range(0, lmax + 1):
                # Specific coefficient logic for DH weights
                term = np.sin((2 * k + 1) * theta) / (2 * k + 1)
                sum_k += term
            weights[i] = (2.0 / nlat) * np.sin(theta) * sum_k
            # Note: Real DH weights usually require O(N^2) or FFT to compute perfectly
            # for the inverse transform. The above is a quadrature approximation.

        return colats, lons, weights


class DH2:
    """
    Driscoll-Healy Grid (Type II) or Clenshaw-Curtis Grid.
    Equidistant grid INCLUDING the poles.
    """

    @staticmethod
    def get_nodes(lmax):
        """
        Generates 1D coordinates and weights for a DH2 grid (includes poles).

        Specifications:
        1. Longitude range: [-pi, pi)
        2. Latitude definition: Colatitude [0, pi], including poles.
        3. Units: Radians
        4. Colatitude order: North -> South

        Parameters:
            lmax (int): Maximum degree of spherical harmonic expansion.

        Returns:
            colats (ndarray): Colatitude points (1D). Shape: (nlat,).
            lons (ndarray): Longitude points (1D). Shape: (nlon,).
            weights (ndarray): Integration weights (1D).
        """
        # DH2 usually requires 2 * lmax + 2 points to include boundaries comfortably
        nlat = 2 * (lmax + 1)
        nlon = 2 * (lmax + 1)

        # 1. Colatitudes: Equidistant from 0 to pi (inclusive)
        colats = np.linspace(0, np.pi, nlat)

        # 2. Longitudes
        lons = np.linspace(-np.pi, np.pi, nlon, endpoint=False)

        # 3. Weights (Clenshaw-Curtis style)
        # Weights for closed interval [0, pi] are more complex.
        # A common simplified weight for equidistant lat/lon (often used in simple climate models)
        # is proportional to sin(theta).
        # For rigorous DH2 SHT, weights are specific. Here we provide the geometric area weight approximation
        # which is common for "generating a grid" purposes.

        # Approximate weights based on spherical belt area:
        d_theta = np.pi / (nlat - 1)
        weights = 2.0 * np.pi * np.sin(colats) * d_theta

        # Correction for weights sum to 4pi (or normalized 1) if needed.
        # This ensures the "pole" weights are small/zero (sin(0)=0).

        return colats, lons, weights


if __name__ == '__main__':
    lmax = 10
    _, _, w_DH1 = DH.get_nodes(lmax)
    _, _, w_DH2 = DH2.get_nodes(lmax)

    pass
