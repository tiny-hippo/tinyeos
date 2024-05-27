from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import UnivariateSpline
from scipy.interpolate.interpnd import NDInterpolatorBase, _ndim_coords_from_arrays
from scipy.spatial import cKDTree
from sklearn.neighbors._base import _get_weights

from tinyeos.definitions import eos_num_vals, eps1, eps2, tiny_val

# constants (in cgs units)
A_h = 1.0078  # atomic mass of hydrogen
A_he = 4.0026  # atomic mass of helium
m_e = 9.10938e-28  # electron mass
m_u = 1.66e-24  # atomic mass unit
m_h = A_h * m_u  # hydrogen (atom) mass
m_he = A_he * m_u  # helium  (atom) mass
k_b = 1.3806503e-16  # boltzmann constant
h = 6.260755e-27  # planck constant
sigma_b = 5.6704e-5  # stefan-boltzmann constant
Na = 6.02214199e23  # avogadro constant
Z_sun = 0.014  # solar heavy-element mass fraction
X_proto = 0.705  # protosolar helium abundance
Y_proto = 0.275  # protosolar hydrogen abundance
proto_ratio = Y_proto / X_proto


class NearestND(NDInterpolatorBase):
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        rescale: bool = False,
        tree_options: bool = None,
    ):
        NDInterpolatorBase.__init__(
            self, x, y, rescale=rescale, need_contiguous=False, need_values=False
        )
        if tree_options is None:
            tree_options = dict()
        self.tree = cKDTree(self.points, **tree_options)
        self.values = np.asarray(y)

    def __call__(self, *args: Tuple, k: int = 1, weights: str = "uniform"):
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)
        dist, i = self.tree.query(xi, k=k)

        if k == 1:
            return self.values[i]

        if xi.ndim == 1:
            which_axis = 0
        elif xi.ndim == 2:
            which_axis = 1
        else:
            msg = "input dimension not supported"
            raise NotImplementedError(msg)

        if weights in [None, "uniform"]:
            return np.average(self.values[i], axis=which_axis)
        else:
            if xi.ndim == 1:
                weight_matrix = dist
            elif xi.ndim == 2:
                weight_matrix = _get_weights(dist, weights=weights)
                weight_matrix = np.repeat(
                    weight_matrix[:, :, np.newaxis], self.values.shape[-1], axis=2
                )
            else:
                msg = "input dimension not supported"
                raise NotImplementedError(msg)

            return np.average(self.values[i], axis=which_axis, weights=weight_matrix)


def get_X(Z: ArrayLike) -> ArrayLike:
    """Calculates the hydrogen mass fraction given
    a heavy-element fraction Z and assuming protosolar
    composition.

    Args:
        Z (float): heavy-element mass fraction

    Returns:
        X (float): hydrogen mass fraction
    """
    X = (1 - Z) / (1 + proto_ratio)
    return X


def get_Z_from_FeH(FeH: ArrayLike) -> ArrayLike:
    """Calculates the heavy-element mass fraction
    given a metallicity.

    Args:
        FeH (ArrayLike): metallicity

    Returns:
        ArrayLike: heavy-element mass fraction
    """
    Z = Z_sun * 10**FeH
    return Z


def get_FeH_from_Z(Z: ArrayLike) -> ArrayLike:
    """Calculates the metallicity given a heavy-
    element mass fraction.

    Args:
        Z (ArrayLike): heavy-element mass fraction

    Returns:
        ArrayLike: metallicity
    """
    Z = max(Z, Z_sun)
    FeH = np.log10(Z / Z_sun)
    return FeH


def get_1d_spline(
    x: ArrayLike, y: ArrayLike, K: int = 1, S: int = 0
) -> UnivariateSpline:
    """Wrapper for UniviarateSpline.

    Args:
        x ((N,) ArrayLike): 1-D array of independent input data.
        y ((N,) ArrayLike): 1-D array of dependent input data.
        K (int, optional): degree of the smoothing spline. Defaults to 1.
        S (int, optional): positive smoothing factor. Defaults to 0.

    Returns:
        UnivariateSpline: fitted 1-D smoothing spline.
    """

    spl = UnivariateSpline(x, y, s=S, k=K, ext=0)
    return spl


def get_zeros(
    input_shape: Tuple[int, ...],
    num_vals: int = eos_num_vals,
    empty: bool = False,
) -> NDArray:
    """Helper function to return a result array of the appropriate shape.

    Args:
        input_shape (Tuple): shape of the inputs.
        num_vals (int, optional): number of values in the first dimension.
            Defaults to num_eos_vals.
        empty (bool, optional): whether to return an empty array.
            Defaults to False.

    Returns:
        NDArray
    """
    input_ndim = len(input_shape)
    shape = (num_vals,) + input_shape if input_ndim > 0 else num_vals
    if empty:
        return np.empty(shape=shape, dtype=np.float64)
    else:
        return np.zeros(shape=shape, dtype=np.float64)


def check_composition(
    X: ArrayLike, Z: ArrayLike
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Checks whether input composition adds up to less than 1,
    and formats the mass fractions.

    Args:
        X (ArrayLike): hydrogen mass fraction.
        Z (ArrayLike): heavy-element mass fraction.

    Raises:
        ValueError: X and Z must have equal shape
            and their sum must be smaller or equal 1.

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike]: tuple of hydrogen,
            helium and heavy-element mass fractions.
    """

    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)

    if X.shape != Z.shape:
        msg = "X and Z must have equal shape"
        raise ValueError(msg)

    if not np.all(X + Z <= 1):
        msg = "X + Z must be smaller than 1"
        raise ValueError(msg)

    Y = 1 - X - Z
    composition = np.asarray([X, Y, Z])
    check_zero = np.isclose(composition, 0, atol=eps1)
    composition[check_zero] = tiny_val
    check_one = np.isclose(composition, 1, atol=eps1)
    composition[check_one] = 1
    X = composition[0]
    Y = composition[1]
    Z = composition[2]
    return (X, Y, Z)


def ideal_mixing_law(
    rho_x: ArrayLike,
    rho_y: ArrayLike,
    rho_z: ArrayLike,
    X: ArrayLike,
    Y: ArrayLike,
    Z: ArrayLike,
) -> ArrayLike:
    """Implementation of the ideal mixing law:
    1 / rho = X / rho_x + Y / rho_y + Z / rho_y

    Args:
        rho_x (ArrayLike): density of hydrogen.
        rho_y (ArrayLike): density of helium.
        rho_z (ArrayLike): density of the heavy element.
        X (ArrayLike): hydrogen mass fraction.
        Y (ArrayLike): helium mass fraction.
        Z (ArrayLike): heavy-element mass fraction.

    Returns:
        ArrayLike: inverse total density.
    """
    if isinstance(rho_x, float):
        max_ndim = 0
    else:
        max_ndim = np.max([rho_x.ndim, X.ndim, Y.ndim, Z.ndim])

    if max_ndim > 0:
        x_rho_x = np.zeros_like(rho_x)
        i = np.logical_and(eps1 < X, rho_x > eps2)
        x_rho_x[i] = X[i] / rho_x[i]

        y_rho_y = np.zeros_like(rho_y)
        i = np.logical_and(eps1 < Y, rho_y > eps2)
        y_rho_y[i] = Y[i] / rho_y[i]

        z_rho_z = np.zeros_like(rho_z)
        i = np.logical_and(eps1 < Z, rho_z > eps2)
        z_rho_z[i] = Z[i] / rho_z[i]
    else:
        if np.isclose(X, 0, atol=eps1) or np.isclose(rho_x, 0, atol=eps2):
            x_rho_x = 0
        else:
            x_rho_x = X / rho_x
        if np.isclose(Y, 0, atol=eps1) or np.isclose(rho_y, 0, atol=eps2):
            y_rho_y = 0
        else:
            y_rho_y = Y / rho_y
        if np.isclose(Z, 0, atol=eps1) or np.isclose(rho_z, 0, atol=eps2):
            z_rho_z = 0
        else:
            z_rho_z = Z / rho_z
    return x_rho_x + y_rho_y + z_rho_z


def get_xy_number_fractions(
    Y: ArrayLike, ionized: bool = False
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculates the hydrogen and helium number fractions
    of a hydrogen-helium mixture.

    Args:
        Y (ArrayLike): helium mass fraction
        ionized (bool, optional): whether the mixture is fully
            ionized or not. Defaults to False.

    Returns:
        Tuple[ArrayLike, ArrayLike]: tuple of hydrogen and helium
            number fractions.
    """
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    X = 1 - Y
    x_h2 = 0  # no molecular hydrogen
    mu = 2 * X + 3 / 4 * Y if ionized else X / (1 + x_h2) + Y / 4
    mu = 1 / mu

    x_he = Y * mu / A_he
    x_h = 1 - x_he
    return (x_h, x_he)


def get_xyz_number_fractions(
    Y: ArrayLike, Z: ArrayLike, A_z: float
) -> Tuple[ArrayLike, ArrayLike]:
    """Calculates the hydrogen, helium and heavy-element number fractions.

    Args:
        Y (ArrayLike): helium mass fraction.
        Z (ArrayLike): heavy-element mass fraction.

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike]: tuple of the number fractions.
    """
    # Y_eff = Y / (1 - Z)
    Y_eff = Y
    N_tot = (1 - Y_eff) * (1 - Z) / A_h + (Y_eff * (1 - Z) / A_he) + Z / A_z

    x_h = (1 - Y_eff) * (1 - Z) / (A_h * N_tot)
    x_z = Z / (A_z * N_tot)
    x_he = 1 - x_h - x_z
    return (x_h, x_he, x_z)


def xlogx(x: ArrayLike) -> ArrayLike:
    """Calculates x * log(x).

    Args:
        x (ArrayLike): input array.

    Returns:
        ArrayLike: x * log(x).
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim > 0:
        xlogx = np.zeros_like(x)
        i = x > 0
        xlogx[i] = x[i] * np.log(x[i])
    else:
        xlogx = 0 if x <= 0 else x * np.log(x)
    return xlogx


def get_mixing_entropy(Y: ArrayLike, Z: ArrayLike, A_z: float) -> ArrayLike:
    """Calculates the ideal mixing entropy of the hydrogen, helium and
    heavy-element mixture (see eq. 11 of Chabrier et al. (2019)).
    Free-electron entropy neglected and the mean molecular weight
    calculation is heavily simplified.

    Args:
        Y (ArrayLike): helium mass-fraction.
        Z (ArrayLike): heavy-element mass-fraction.
        A_z (float): atomic mass of the heavy element.

    Returns:
        ArrayLike: ideal mixing entropy.
    """
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    if not isinstance(Z, np.ndarray):
        Z = np.array(Z)
    x = np.zeros((3,) + Y.shape)  # number fractions
    i = Z > 0
    x[0, i], x[1, i], x[2, i] = get_xyz_number_fractions(Y=Y[i], Z=Z[i], A_z=A_z)
    x[0, ~i], x[1, ~i] = get_xy_number_fractions(Y=Y[~i])

    A_mean = x[0] * A_h + x[1] * A_he + x[2] * A_z
    S_mix = -k_b * (xlogx(x[0]) + xlogx(x[1]) + xlogx(x[2])) / (A_mean * m_u)
    return S_mix


def get_eta(logT: ArrayLike, logRho: ArrayLike, log_free_e: ArrayLike) -> ArrayLike:
    """Calculates the inverse electron chemical potential
    by inverting the fermi integral using the
    rational function approximation for Fermi-Dirac
    integrals (Antia, 1993)

    Args:
        logT (ArrayLike): log10 of temperature.
        logRho (ArrayLike): log10 of density.
        log_free_e (ArrayLike): log10 of mean number of free electrons
            per nucleon (inverse electron mean molecular weight).

    Raises:
        ValueError: logT, logRho and log_free_e must
            have equal shape.

    Returns:
        ArrayLike: inverse electron chemical potential
    """

    if not isinstance(logT, np.ndarray):
        logT = np.array(logT)
    if not isinstance(logRho, np.ndarray):
        logRho = np.array(logRho)
    if not isinstance(log_free_e, np.ndarray):
        log_free_e = np.array(log_free_e)

    if (
        logT.shape != logRho.shape
        or logT.shape != log_free_e.shape
        or logRho.shape != log_free_e.shape
    ):
        msg = "input must have the same shape"
        raise ValueError(msg)

    T = 10**logT
    fac = (4 * np.pi / pow(h, 3)) * pow((2 * m_e * k_b * T), 1.5)
    log_ne = log_free_e + np.log10(Na) + logRho
    n_e = 10**log_ne  # Free electron number density
    f = n_e / fac  # Fermi integral

    an = 0.5
    m1, k1, m2, k2 = 1, 1, 1, 1
    a1 = np.array([4.4593646e1, 1.1288764e1, 1.0])
    b1 = np.array([3.9519346e1, -5.7517464e0, 2.6594291e-1])
    a2 = np.array([3.4873722e1, -2.6922515e1, 1.0])
    b2 = np.array([2.6612832e1, -2.0452930e1, 1.1808945e1])

    if logT.ndim > 0:
        eta = np.zeros_like(f)
        i = f < 4
        rn = f[i] + a1[m1]
        rn = rn * f[i] + a1[m1 - 1]
        den = b1[k1 + 1]
        den = den * f[i] + b1[k1]
        den = den * f[i] + b1[k1 - 1]
        eta[i] = np.log(f[i] * rn / den)

        i = ~i
        ff = 1.0 / f[i] ** (1.0 / (1 + an))
        rn = ff + a2[m2]
        rn = rn * ff + a2[m2 - 1]
        den = b2[k2 + 1]
        den = den * ff + b2[k2]
        den = den * ff + b2[k2 - 1]
        eta[i] = rn / (den * ff)
        eta[eta > 999] = 999
        eta[eta < -999] = -999
    else:
        if f < 4:
            rn = f + a1[m1]
            rn = rn * f + a1[m1 - 1]
            den = b1[k1 + 1]
            for i in range(k1, -1, -1):
                den = den * f + b1[i]
            eta = np.log(f * rn / den)
        else:
            ff = 1.0 / f ** (1.0 / (1 + an))
            rn = ff + a2[m2]
            rn = rn * ff + a2[m2 - 1]
            den = b2[k2 + 1]
            for i in range(k2, -1, -1):
                den = den * ff + b2[i]
            eta = rn / (den * ff)
        if eta > 999:
            eta = 999
        elif eta < -999:
            eta = -999

    return eta
