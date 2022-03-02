import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.interpolate import UnivariateSpline
from scipy.interpolate.interpnd import NDInterpolatorBase, _ndim_coords_from_arrays
from sklearn.neighbors._base import _get_weights

# constants
m_e = 9.10938e-28  # electron mass in g
k_b = 1.3806503e-16  # boltzmann constant in erg/K
h = 6.260755e-27  # planck constant in erg s
Na = 6.02214199e23  # avogadro's constant
Zsun = 0.014  # solar heavy-element mass fraction
Xproto = 0.705  # protosolar helium abundance
Yproto = 0.275  # protosolar hydrogen abundance
proto_ratio = Yproto / Xproto


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
        Z (float): Heavy-element mass fraction

    Returns:
        X (float): Hydrogen mass fraction
    """
    X = (1 - Z) / (1 + proto_ratio)
    return X


def get_Z_from_FeH(FeH: ArrayLike) -> ArrayLike:
    """Calculates the heavy-element mass fraction
    given a metallicity.

    Args:
        FeH (ArrayLike): Metallicity

    Returns:
        ArrayLike: Heavy-element mass fraction
    """
    Z = Zsun * 10**FeH
    return Z


def get_FeH_from_Z(Z: float) -> float:
    """Calculates the metallicity given a heavy-
    element mass fraction.

    Args:
        Z (float): Heavy-element mass fraction

    Returns:
        float: Metallicity
    """
    Z = max(Z, Zsun)
    FeH = np.log10(Z / Zsun)
    return FeH


def get_1d_spline(
    x: ArrayLike, y: ArrayLike, K: int = 1, S: int = 0
) -> UnivariateSpline:
    """Wrapper for UniviarateSpline.

    Args:
        x ((N,) array_like): 1-D array of independent input data.
        y ((N,) array_like): 1-D array of dependent input data.
        K (int, optional): Degree of the smoothing spline. Defaults to 1.
        S (int, optional): Positive smoothing factor. Defaults to 0.

    Returns:
        UnivariateSpline: Fitted 1-D smoothing spline.
    """

    spl = UnivariateSpline(x, y, s=S, k=K, ext=0)
    return spl


def check_composition(X: float, Z: float) -> Tuple:
    """Checks whether input composition adds up to less than 1,
    and formats the mass fractions.


    Args:
        X (float): Hydrogen mass fraction.
        Z (float): Heavy-element mass fraction.

    Returns:
        tuple: Tuple of hydrogen, helium and heavy-element mass fractions.
    """

    assert np.all(X + Z <= 1), "X + Z must be smaller than 1"
    eps = 1e-3
    Y = 1 - X - Z
    composition = np.array([X, Y, Z])
    check_zero = np.isclose(composition, 0, atol=eps)
    composition[check_zero] = 0
    check_one = np.isclose(composition, 1, atol=eps)
    composition[check_one] = 1
    X = composition[0]
    Y = composition[1]
    Z = composition[2]
    return (X, Y, Z)


def ideal_mixing_law(
    rho_x: float, rho_y: float, rho_z: float, X: float, Y: float, Z: float
) -> float:
    """Implementation of the ideal mixing law:
    1 / rho = X / rho_x + Y / rho_y + Z / rho_y

    Args:
        rho_x (float): Density of hydrogen.
        rho_y (float): Density of helium.
        rho_z (float): Density of the heavy element.
        X (float): Hydrogen mass fraction.
        Y (float): Helium mass fraction.
        Z (float): Heavy-element mass fraction.

    Returns:
        float: Inverse total density.
    """

    eps = 1e-15
    if X == 0 or np.isclose(rho_x, 0, atol=eps):
        x_rho_x = 0
    else:
        x_rho_x = X / rho_x
    if Y == 0 or np.isclose(rho_y, 0, atol=eps):
        y_rho_y = 0
    else:
        y_rho_y = Y / rho_y
    if Z == 0 or np.isclose(rho_z, 0, atol=eps):
        z_rho_z = 0
    else:
        z_rho_z = Z / rho_z

    return x_rho_x + y_rho_y + z_rho_z


def get_eta(logT: float, logRho: float, log_free_e: float) -> float:
    """Calculates the inverse electron chemical potential
    by inverting the fermi integral using the
    rational function approximation for Fermi-Dirac
    integrals (Antia, 1993)

    Args:
        logT (float): Log10 of temperature.
        logRho (float): Log10 of density.
        log_free_e ([type]): Log10 of mean number of free electrons
        per nucleon (inverse electron mean molecular weight).

    Returns:
        float: Inverse electron chemical potential
    """

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
