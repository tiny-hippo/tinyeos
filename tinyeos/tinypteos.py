import os
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike, NDArray
from pathlib import Path
from tinyeos.interpolantsbuilder import InterpolantsBuilder
from tinyeos.support import get_eta, ideal_mixing_law, check_composition
from tinyeos.definitions import *


class TinyPT(InterpolantsBuilder):
    """Temperature-pressure equation of state for a mixture of hydrogen,
    helium and a heavy element. Units are cgs everywhere.

    Equations of state implemented:
        Hydrogen-Helium:
            CMS (Chabrier et al. 2019),
            SCvH (Saumon et al. 1995).

        Heavy-Element:
            Water (QEoS from More et al. 1988),
            Water (AQUA from Haldemann et al. 2020),
            SiO2 (QEoS, More et al. 1988),
            Mixture of water and rock (QEoS),
            Iron (QEoS, more et al. 1998).
    """

    def __init__(
        self,
        which_heavy: str = "water",
        which_hhe: str = "cms",
        build_interpolants: bool = False,
    ) -> None:
        """__init__ method. Defines parameters and either loads or
        builds the interpolants.

        Args:
            which_heavy (str, optional): which heavy-element equation of state
            to use. Defaults to "water". Options are "water", "rock",
            "mixture", "aqua" or "iron".
            which_hhe (str, optional): which hydrogen-helium equation of state
            to use. Defaults to "cms". Options are "cms" or "scvh".
            build_interpolants (bool, optional): whether to build interpolants.
            Defaults to False.

        Raises:
            NotImplementedError: raised if which_heavy or which_hhe choices
            are unavailable.
        """

        if build_interpolants:
            super().__init__()

        self.logP_max = logP_max
        self.logP_min = logP_min
        self.logT_max = logT_max
        self.logT_min = logT_min
        self.logRho_max = logRho_max
        self.logRho_min = logRho_min
        self.eps = 1e-3

        self.num_vals = num_vals
        self.i_logT = i_logT
        self.i_logRho = i_logRho
        self.i_logP = i_logP
        self.i_logS = i_logS
        self.i_logU = i_logU
        self.i_chiRho = i_chiRho
        self.i_chiT = i_chiT
        self.i_grad_ad = i_grad_ad
        self.i_cp = i_cp
        self.i_cv = i_cv
        self.i_gamma1 = i_gamma1
        self.i_gamma3 = i_gamma3
        self.i_dS_dT = i_dS_dT
        self.i_dS_dRho = i_dS_dRho
        self.i_dE_dRho = i_dE_dRho
        self.i_mu = i_mu
        self.i_eta = i_eta
        self.i_lfe = i_lfe
        self.i_csound = i_csound

        self.kwargs = {"grid": False}
        self.cache_path = Path(__file__).parent / "data/eos/interpolants"
        if which_heavy not in ["water", "rock", "aqua", "mixture"]:
            raise NotImplementedError("Invalid option for which_heavy")
        if which_hhe not in ["cms", "scvh"]:
            raise NotImplementedError("Invalid option for which_hhe")

        # Heavy element: Charge, Atomic Mass
        # h2o: 10, 18.015
        # sio2: 30, 60.080
        # fe: 26, 55.845
        self.heavy_element = which_heavy
        if which_heavy == "water" or which_heavy == "aqua":
            self.A = 18.015
        elif which_heavy == "rock":
            self.A = 60.080
        elif which_heavy == "iron":
            self.A = 55.845
        elif which_heavy == "mixture":
            self.A = 0.5 * (18.015 + 60.080)
        else:
            raise NotImplementedError("Invalid option for which_heavy.")

        self.interpPT_x = self.__load_interp("interpPT_x_" + which_hhe + ".npy")
        self.interpPT_y = self.__load_interp("interpPT_y_" + which_hhe + ".npy")
        self.interpPT_z = self.__load_interp("interpPT_z_" + which_heavy + ".npy")
        self.interpDT_z = self.__load_interp("interpDT_z_" + which_heavy + ".npy")

        self.interpPT_logRho_x = self.interpPT_x[0]
        self.interpPT_logS_x = self.interpPT_x[1]
        self.interpPT_logU_x = self.interpPT_x[2]
        self.interpPT_dlRho_dlT_P_x = self.interpPT_x[3]
        self.interpPT_dlRho_dlP_T_x = self.interpPT_x[4]
        self.interpPT_dlS_dlT_P_x = self.interpPT_x[5]
        self.interpPT_dlS_dlP_T_x = self.interpPT_x[6]
        self.interpPT_grad_ad_x = self.interpPT_x[7]
        self.interpPT_lfe_x = self.interpPT_x[8]
        self.interpPT_mu_x = self.interpPT_x[9]

        self.interpPT_logRho_y = self.interpPT_y[0]
        self.interpPT_logS_y = self.interpPT_y[1]
        self.interpPT_logU_y = self.interpPT_y[2]
        self.interpPT_dlRho_dlT_P_y = self.interpPT_y[3]
        self.interpPT_dlRho_dlP_T_y = self.interpPT_y[4]
        self.interpPT_dlS_dlT_P_y = self.interpPT_y[5]
        self.interpPT_dlS_dlP_T_y = self.interpPT_y[6]
        self.interpPT_grad_ad_y = self.interpPT_y[7]
        self.interpPT_lfe_y = self.interpPT_y[8]
        self.interpPT_mu_y = self.interpPT_y[9]

        self.interpPT_logRho_z = self.interpPT_z[0]
        self.interpPT_logS_z = self.interpPT_z[1]
        self.interpPT_logU_z = self.interpPT_z[2]
        if which_heavy == "aqua":
            self.interpPT_grad_ad_z = self.interpPT_z[3]

        self.interpDT_logP_z = self.interpDT_z[0]
        self.interpDT_logS_z = self.interpDT_z[1]

    def __call__(
        self, logT: ArrayLike, logP: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> NDArray:
        """__call__ method acting as convenience wrapper for the evaluate method.
        Calculates the equation of state output for the mixture.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.
            X (ArrayLike): hydrogen mass-fraction.
            Z (ArrayLike): heavy-element mass-fraction.

        Returns:
            NDArray: Equation of state output. The index of the individual
            quantities is defined in the __init__ method.
        """
        return self.evaluate(logT, logP, X, Z)

    def __load_interp(self, filename: str) -> object:
        """Loads the interpolant from the disk.

        Args:
            filename (str): name of the interpolant cache file.

        Raises:
            FileNotFoundError: raised if the interpolant was not found.

        Returns:
            object: bivariate spline loaded from the cache file.
        """
        src = os.path.join(self.cache_path, filename)
        if not os.path.isfile(src):
            raise FileNotFoundError("Missing interpolant cache " + src)
        return np.load(src, allow_pickle=True)

    def __check_PT(self, logT: ArrayLike, logP: ArrayLike) -> Tuple[NDArray, NDArray]:
        """Makes sure that input temperature and pressure
        are within equation of state limits.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Raises:
            ValueError: logT and logP must have equal shape
            and all values must be within the equation of
            state limits.

        Returns:
            Tuple[NDArray, NDArray]: (logT, logP) as arrays.
        """

        if not isinstance(logT, np.ndarray):
            logT = np.array(logT)
        if not isinstance(logP, np.ndarray):
            logP = np.array(logP)

        if not logT.shape == logP.shape:
            msg = "logT and logP must have equal shape"
            raise ValueError(msg)
        if np.any(logT < self.logT_min) or np.any(logT > self.logT_max):
            msg = "logT out of bounds"
            raise ValueError(msg)
        elif np.any(logP < self.logP_min) or np.any(logP > self.logP_max):
            msg = "logP out of bounds"
            raise ValueError(msg)
        else:
            return (logT, logP)

    def __get_zeros(
        self,
        logT: NDArray,
        logP: NDArray,
        X: NDArray = np.array(0),
        Z: NDArray = np.array(0),
    ) -> NDArray:
        """Helper function to return a result array of the appropriate shape

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.
            X (ArrayLike): hydrogen mass-fraction.
            Z (ArrayLike): heavy-element mass fraction.

        Raises:
            ValueError: input can at most be two-dimensional.

        Returns:
            NDArray
        """
        max_ndim = np.max([logT.ndim, logP.ndim, X.ndim, Z.ndim])
        if max_ndim > 0:
            shape = (self.num_vals,) + logT.shape
            return np.zeros(shape)
        elif max_ndim == 0:
            return np.zeros(self.num_vals)
        else:
            msg = "unsupported input shape"
            raise ValueError(msg)

    def __ideal_mixture(
        self, logT: ArrayLike, logP: ArrayLike, X: ArrayLike, Y: ArrayLike, Z: ArrayLike
    ) -> ArrayLike:
        """Calculates the total density of the gas using the ideal
        mixing law.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.
            X (ArrayLike): hydrogen mass-fraction
            Y (ArrayLike): helium mass-fraction.
            Z (ArrayLike): heavy-element mass-fraction.

        Returns:
            ArrayLike: total density of the mixure.
        """

        if np.all(X == 1):
            logRho = self.interpPT_logRho_x(logT, logP, grid=False)
        elif np.all(Y == 1):
            logRho = self.interpPT_logRho_y(logT, logP, grid=False)
        elif np.all(Z == 1):
            logRho = self.interpPT_logRho_z(logT, logP, grid=False)
        else:
            logRho_x = self.interpPT_logRho_x(logT, logP, grid=False)
            logRho_y = self.interpPT_logRho_y(logT, logP, grid=False)
            logRho_z = self.interpPT_logRho_z(logT, logP, grid=False)

            iml = ideal_mixing_law(
                10**logRho_x, 10**logRho_y, 10**logRho_z, X, Y, Z
            )
            logRho = np.log10(1 / iml)

        return logRho

    def __evaluate_x(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for hydrogen.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """

        logRho = self.interpPT_logRho_x(logT, logP, **self.kwargs)
        logS = self.interpPT_logS_x(logT, logP, **self.kwargs)
        logU = self.interpPT_logU_x(logT, logP, **self.kwargs)

        dlRho_dlP_T = self.interpPT_dlRho_dlP_T_x(logT, logP, **self.kwargs)
        dlRho_dlT_P = self.interpPT_dlRho_dlT_P_x(logT, logP, **self.kwargs)

        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        grad_ad = self.interpPT_grad_ad_x(logT, logP, **self.kwargs)
        lfe = self.interpPT_lfe_x(logT, logP, **self.kwargs)
        mu = self.interpPT_mu_x(logT, logP, **self.kwargs)

        res_x = self.__get_zeros(logT, logP)
        res_x[self.i_logT] = logT
        res_x[self.i_logRho] = logRho
        res_x[self.i_logP] = logP
        res_x[self.i_logS] = logS
        res_x[self.i_logU] = logU
        res_x[self.i_chiRho] = chiRho
        res_x[self.i_chiT] = chiT
        res_x[self.i_grad_ad] = grad_ad
        res_x[self.i_mu] = mu
        res_x[self.i_lfe] = lfe

        return res_x

    def __evaluate_y(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for helium.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """

        logRho = self.interpPT_logRho_y(logT, logP, **self.kwargs)
        logS = self.interpPT_logS_y(logT, logP, **self.kwargs)
        logU = self.interpPT_logU_y(logT, logP, **self.kwargs)

        dlRho_dlP_T = self.interpPT_dlRho_dlP_T_y(logT, logP, **self.kwargs)
        dlRho_dlT_P = self.interpPT_dlRho_dlT_P_y(logT, logP, **self.kwargs)

        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        grad_ad = self.interpPT_grad_ad_y(logT, logP, **self.kwargs)
        lfe = self.interpPT_lfe_y(logT, logP, **self.kwargs)
        mu = self.interpPT_mu_y(logT, logP, **self.kwargs)

        res_y = self.__get_zeros(logT, logP)
        res_y[self.i_logT] = logT
        res_y[self.i_logRho] = logRho
        res_y[self.i_logP] = logP
        res_y[self.i_logS] = logS
        res_y[self.i_logU] = logU
        res_y[self.i_chiRho] = chiRho
        res_y[self.i_chiT] = chiT
        res_y[self.i_grad_ad] = grad_ad
        res_y[self.i_mu] = mu
        res_y[self.i_lfe] = lfe

        return res_y

    def __evaluate_z(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for the heavy element..

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """
        logRho = self.interpPT_logRho_z(logT, logP, **self.kwargs)
        logS = self.interpPT_logS_z(logT, logP, **self.kwargs)
        logU = self.interpPT_logU_z(logT, logP, **self.kwargs)

        chiRho = self.interpDT_logP_z(logT, logRho, dy=1, **self.kwargs)
        chiT = self.interpDT_logP_z(logT, logRho, dx=1, **self.kwargs)
        dlS_dlT = self.interpDT_logS_z(logT, logRho, dx=1, **self.kwargs)
        dlS_dlRho = self.interpDT_logS_z(logT, logRho, dy=1, **self.kwargs)

        if self.heavy_element == "aqua":
            grad_ad = self.interpPT_grad_ad_z(logT, logP, **self.kwargs)
        else:
            grad_ad = 1 / (chiT - dlS_dlT * chiRho / dlS_dlRho)

        res_z = self.__get_zeros(logT, logP)
        res_z[self.i_logT] = logT
        res_z[self.i_logRho] = logRho
        res_z[self.i_logP] = logP
        res_z[self.i_logS] = logS
        res_z[self.i_logU] = logU
        res_z[self.i_chiRho] = chiRho
        res_z[self.i_chiT] = chiT
        res_z[self.i_grad_ad] = grad_ad
        res_z[self.i_mu] = self.A  # atomic weight
        res_z[self.i_lfe] = -99  # not available in tables
        res_z[self.i_eta] = 999  # not available since lfe is missing

        return res_z

    def evaluate_legacy(self, logT: float, logP: float, X: float, Z: float) -> NDArray:
        """Calculates the equation of state output for the mixture.

        Args:
            logT (float): log10 of the temperature.
            logP (float): log10 of the pressure.
            X (float): hydrogen mass-fraction.
            Z (float): heavy-element mass-fraction.

        Returns:
            NDArray: equation of state output. The index of the individual
            quantities is defined in the __init__ method.
        """

        logT, logP = self.__check_PT(logT, logP)
        X, Y, Z = check_composition(X, Z)

        if X > 0:
            res_x = self.__evaluate_x(logT, logP)
        else:
            res_x = np.zeros(self.num_vals)
        if Y > 0:
            res_y = self.__evaluate_y(logT, logP)
        else:
            res_y = np.zeros(self.num_vals)
        if Z > 0:
            res_z = self.__evaluate_z(logT, logP)
        else:
            res_z = np.zeros(self.num_vals)

        logRho = self.__ideal_mixture(logT, logP, X, Y, Z)
        logRho_x = res_x[self.i_logRho]
        logRho_y = res_y[self.i_logRho]
        logRho_z = res_z[self.i_logRho]

        T = 10**logT
        P = 10**logP
        rho = 10**logRho
        rho_x = 10**logRho_x
        rho_y = 10**logRho_y
        rho_z = 10**logRho_z

        if X == 1:
            logRho = res_x[self.i_logRho]
            logS = res_x[self.i_logS]
            logU = res_x[self.i_logU]
            chiRho = res_x[self.i_chiRho]
            chiT = res_x[self.i_chiT]
            grad_ad = res_x[self.i_grad_ad]
            if grad_ad < 0:
                grad_ad = 0
            if chiRho < 0:
                chiRho = 0
            if chiT < 0:
                chiT = 0
            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            if chiT == 0 or gamma3 == 1:
                dlS_dlT_P = self.interpPT_dlS_dlT_P_x(logT, logP, **self.kwargs)
                cp = 10**logS * dlS_dlT_P
            else:
                cp = chiT * P / (rho * T * (gamma3 - 1))
            if chiRho == 0:
                cv = cp
            else:
                cv = cp + P + chiT**2 / (rho * T * chiRho)
            dS_dT = cv / T
            dS_dRho = -(P / T / rho**2) * chiT
            dE_dRho = (P / rho**2) * (1 - chiT)
            lfe = res_x[self.i_lfe]
            mu = res_x[self.i_mu]
            eta = get_eta(logT, logRho, lfe)

        elif Y == 1:
            logRho = res_y[self.i_logRho]
            logS = res_y[self.i_logS]
            logU = res_y[self.i_logU]
            chiRho = res_y[self.i_chiRho]
            chiT = res_y[self.i_chiT]
            grad_ad = res_y[self.i_grad_ad]
            if grad_ad < 0:
                grad_ad = 0
            if chiRho < 0:
                chiRho = 0
            if chiT < 0:
                chiT = 0
            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            if chiT == 0 or gamma3 == 1:
                dlS_dlT_P = self.interpPT_dlS_dlT_P_y(logT, logP, **self.kwargs)
                cp = 10**logS * dlS_dlT_P
            else:
                cp = chiT * P / (rho * T * (gamma3 - 1))
            if chiRho == 0:
                cv = cp
            else:
                cv = cp + P + chiT**2 / (rho * T * chiRho)
            dS_dT = cv / T
            dS_dRho = -(P / T / rho**2) * chiT
            dE_dRho = (P / rho**2) * (1 - chiT)
            lfe = res_y[self.i_lfe]
            mu = res_y[self.i_mu]
            eta = get_eta(logT, logRho, lfe)

        elif Z == 1:
            logRho = res_z[self.i_logRho]
            logS = res_z[self.i_logS]
            logU = res_z[self.i_logU]
            chiRho = res_z[self.i_chiRho]
            chiT = res_z[self.i_chiT]
            grad_ad = res_z[self.i_grad_ad]
            if grad_ad < 0:
                grad_ad = 0
            if chiRho < 0:
                chiRho = 0
            if chiT < 0:
                chiT = 0
            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            if chiT == 0 or gamma3 == 1:
                dlS_dlT_P = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)
                cp = 10**logS * dlS_dlT_P
            else:
                cp = chiT * P / (rho * T * (gamma3 - 1))
            if chiRho == 0:
                cv = cp
            else:
                cv = cp + P + chiT**2 / (rho * T * chiRho)
            dS_dT = cv / T
            dS_dRho = -(P / T / rho**2) * chiT
            dE_dRho = (P / rho**2) * (1 - chiT)
            lfe = res_z[self.i_lfe]
            mu = res_z[self.i_mu]
            eta = res_z[self.i_eta]

        elif X == 0:
            logS_y = res_y[self.i_logS]
            logS_z = res_z[self.i_logS]
            S_y = 10**logS_y
            S_z = 10**logS_z
            S = Y * S_y + Z * S_z
            logS = np.log10(S)

            logU_y = res_y[self.i_logU]
            logU_z = res_z[self.i_logU]
            U = Y * (10**logU_y) + Z * (10**logU_z)
            logU = np.log10(U)

            dlS_dlP_T_y = self.interpPT_dlS_dlP_T_y(logT, logP, **self.kwargs)
            dlS_dlT_P_y = self.interpPT_dlS_dlT_P_y(logT, logP, **self.kwargs)
            dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
            dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

            dlS_dlP_T = (Y * S_y * dlS_dlP_T_y + Z * S_z * dlS_dlP_T_z) / S
            dlS_dlT_P = (Y * S_y * dlS_dlT_P_y + Z * S_z * dlS_dlT_P_z) / S

            dlRho_dlP_T = rho * (
                Y / rho_y / res_y[self.i_chiRho] + Z / rho_z / res_z[self.i_chiRho]
            )
            dlRho_dlT_P = rho * (
                Y / rho_y * (-res_y[self.i_chiT] / res_y[self.i_chiRho])
                + Z / rho_z * (-res_z[self.i_chiT] / res_z[self.i_chiRho])
            )

            grad_ad = -dlS_dlP_T / dlS_dlT_P
            if grad_ad < 0:
                grad_ad = 0
            chiRho = 1 / dlRho_dlP_T
            if chiRho < 0:
                chiRho = 0
            chiT = -dlRho_dlT_P / dlRho_dlP_T
            if chiT < 0:
                chiT = 0
            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            cp = S * dlS_dlT_P
            if chiRho == 0:
                cv = cp
            else:
                cv = cp * chiRho / gamma1
            dS_dT = cv / T
            dS_dRho = -(P / T / rho**2) * chiT
            dE_dRho = (P / rho**2) * (1 - chiT)

            lfe = res_y[self.i_lfe]
            mu = 1 / (Y / res_y[self.i_mu] + Z / res_z[self.i_mu])
            eta = get_eta(logT, logRho, lfe)

        elif Y == 0:
            logS_x = res_x[self.i_logS]
            logS_z = res_z[self.i_logS]
            S_x = 10**logS_x
            S_z = 10**logS_z
            S = X * S_x + Z * S_z
            logS = np.log10(S)

            logU_x = res_x[self.i_logU]
            logU_z = res_z[self.i_logU]
            U = X * (10**logU_x) + Z * (10**logU_z)
            logU = np.log10(U)

            dlS_dlP_T_x = self.interpPT_dlS_dlP_T_x(logT, logP, **self.kwargs)
            dlS_dlT_P_x = self.interpPT_dlS_dlT_P_x(logT, logP, **self.kwargs)
            dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
            dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

            dlS_dlP_T = (X * S_x * dlS_dlP_T_x + Z * S_z * dlS_dlP_T_z) / S
            dlS_dlT_P = (X * S_x * dlS_dlT_P_x + Z * S_z * dlS_dlT_P_z) / S

            dlRho_dlP_T = rho * (
                X / rho_x / res_x[self.i_chiRho] + Z / rho_z / res_z[self.i_chiRho]
            )
            dlRho_dlT_P = rho * (
                X / rho_x * (-res_x[self.i_chiT] / res_x[self.i_chiRho])
                + Z / rho_z * (-res_z[self.i_chiT] / res_z[self.i_chiRho])
            )

            grad_ad = -dlS_dlP_T / dlS_dlT_P
            if grad_ad < 0:
                grad_ad = 0
            chiRho = 1 / dlRho_dlP_T
            if chiRho < 0:
                chiRho = 0
            chiT = -dlRho_dlT_P / dlRho_dlP_T
            if chiT < 0:
                chiT = 0
            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            cp = S * dlS_dlT_P
            if chiRho == 0:
                cv = cp
            else:
                cv = cp * chiRho / gamma1
            dS_dT = cv / T
            dS_dRho = -(P / T / rho**2) * chiT
            dE_dRho = (P / rho**2) * (1 - chiT)

            lfe = res_x[self.i_lfe]
            mu = 1 / (X / res_x[self.i_mu] + Z / res_z[self.i_mu])
            eta = get_eta(logT, logRho, lfe)

        elif Z == 0:
            logS_x = res_x[self.i_logS]
            logS_y = res_y[self.i_logS]
            S_x = 10**logS_x
            S_y = 10**logS_y
            S = X * S_x + Y * S_y
            logS = np.log10(S)

            logU_x = res_x[self.i_logU]
            logU_y = res_y[self.i_logU]
            U = X * (10**logU_x) + Y * (10**logU_y)
            logU = np.log10(U)

            dlS_dlP_T_x = self.interpPT_dlS_dlP_T_x(logT, logP, **self.kwargs)
            dlS_dlT_P_x = self.interpPT_dlS_dlT_P_x(logT, logP, **self.kwargs)
            dlS_dlP_T_y = self.interpPT_dlS_dlP_T_y(logT, logP, **self.kwargs)
            dlS_dlT_P_y = self.interpPT_dlS_dlT_P_y(logT, logP, **self.kwargs)

            dlS_dlP_T = (X * S_x * dlS_dlP_T_x + Y * S_y * dlS_dlP_T_y) / S
            dlS_dlT_P = (X * S_x * dlS_dlT_P_x + Y * S_y * dlS_dlT_P_y) / S

            dlRho_dlP_T = rho * (
                X / rho_x / res_x[self.i_chiRho] + Y / rho_y / res_y[self.i_chiRho]
            )
            dlRho_dlT_P = rho * (
                X / rho_x * (-res_x[self.i_chiT] / res_x[self.i_chiRho])
                + Y / rho_y * (-res_y[self.i_chiT] / res_y[self.i_chiRho])
            )

            grad_ad = -dlS_dlP_T / dlS_dlT_P
            if grad_ad < 0:
                grad_ad = 0
            chiRho = 1 / dlRho_dlP_T
            if chiRho < 0:
                chiRho = 0
            chiT = -dlRho_dlT_P / dlRho_dlP_T
            if chiT < 0:
                chiT = 0
            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            cp = S * dlS_dlT_P
            if chiRho == 0:
                cv = cp
            else:
                cv = cp * chiRho / gamma1
            dS_dT = cv / T
            dS_dRho = -(P / T / rho**2) * chiT
            dE_dRho = (P / rho**2) * (1 - chiT)

            lfe = np.log10(X * 10 ** res_x[self.i_lfe] + Y * 10 ** res_y[self.i_lfe])
            mu = 1 / (X / res_x[self.i_mu] + Y / res_y[self.i_mu])
            eta = get_eta(logT, logRho, lfe)

        else:
            # to-do: add mixing entropy
            logS_x = res_x[self.i_logS]
            logS_y = res_y[self.i_logS]
            logS_z = res_z[self.i_logS]
            S_x = 10**logS_x
            S_y = 10**logS_y
            S_z = 10**logS_z
            S = X * S_x + Y * S_y + Z * S_z
            logS = np.log10(S)

            logU_x = res_x[self.i_logU]
            logU_y = res_y[self.i_logU]
            logU_z = res_z[self.i_logU]
            U = X * (10**logU_x) + Y * (10**logU_y) + Z * (10**logU_z)
            logU = np.log10(U)

            dlS_dlP_T_x = self.interpPT_dlS_dlP_T_x(logT, logP, **self.kwargs)
            dlS_dlT_P_x = self.interpPT_dlS_dlT_P_x(logT, logP, **self.kwargs)
            dlS_dlP_T_y = self.interpPT_dlS_dlP_T_y(logT, logP, **self.kwargs)
            dlS_dlT_P_y = self.interpPT_dlS_dlT_P_y(logT, logP, **self.kwargs)
            dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
            dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

            dlS_dlP_T = (
                X * S_x * dlS_dlP_T_x + Y * S_y * dlS_dlP_T_y + Z * S_z * dlS_dlP_T_z
            ) / S
            dlS_dlT_P = (
                X * S_x * dlS_dlT_P_x + Y * S_y * dlS_dlT_P_y + Z * S_z * dlS_dlT_P_z
            ) / S

            dlRho_dlP_T = rho * (
                X / rho_x / res_x[self.i_chiRho]
                + Y / rho_y / res_y[self.i_chiRho]
                + Z / rho_z / res_z[self.i_chiRho]
            )
            dlRho_dlT_P = rho * (
                X / rho_x * (-res_x[self.i_chiT] / res_x[self.i_chiRho])
                + Y / rho_y * (-res_y[self.i_chiT] / res_y[self.i_chiRho])
                + Z / rho_z * (-res_z[self.i_chiT] / res_z[self.i_chiRho])
            )

            # fix interpolation errors
            grad_ad = -dlS_dlP_T / dlS_dlT_P
            if grad_ad < 0:
                grad_ad = 0
            chiRho = 1 / dlRho_dlP_T
            if chiRho < 0:
                chiRho = 0
            chiT = -dlRho_dlT_P / dlRho_dlP_T
            if chiT < 0:
                chiT = 0
            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            c_sound = np.sqrt(P / rho * gamma1)
            cp = S * dlS_dlT_P
            if chiRho == 0:
                cv = cp
            else:
                cv = cp * chiRho / gamma1
            # in mesa:
            # cp = chiT * P / (rho * T * (gamma3 - 1))
            # cv = cp + p + chiT**2 / (rho * T * chiRho)

            # these are at constant density or temperature
            dS_dT = cv / T  # definition of specific heat
            dS_dRho = -(P / T / rho**2) * chiT  # maxwell relation
            dE_dRho = (P / rho**2) * (1 - chiT)
            # 1st law, def. of entropy, total derivative, and maxwell relation

            # only hydrogen and helium contribute to free electrons
            lfe = np.log10(X * 10 ** res_x[self.i_lfe] + Y * 10 ** res_y[self.i_lfe])
            eta = get_eta(logT, logRho, lfe)

            # crude estimate for mean molecular weight of the heavy element
            mu = 1 / (
                X / res_x[self.i_mu] + Y / res_y[self.i_mu] + Z / res_z[self.i_mu]
            )

        res = np.zeros(self.num_vals)
        res[self.i_logT] = logT
        res[self.i_logRho] = logRho
        res[self.i_logP] = logP
        res[self.i_logS] = logS
        res[self.i_logU] = logU
        res[self.i_chiRho] = chiRho
        res[self.i_chiT] = chiT
        res[self.i_grad_ad] = grad_ad
        res[self.i_cp] = cp
        res[self.i_cv] = cv
        res[self.i_gamma1] = gamma1
        res[self.i_gamma3] = gamma3
        res[self.i_dS_dT] = dS_dT
        res[self.i_dS_dRho] = dS_dRho
        res[self.i_dE_dRho] = dE_dRho
        res[self.i_mu] = mu
        res[self.i_eta] = eta
        res[self.i_lfe] = lfe
        res[self.i_csound] = c_sound

        return res

    def evaluate(
        self, logT: ArrayLike, logP: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> NDArray:
        """Calculates the equation of state output for the mixture.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.
            X (ArrayLike): hydrogen mass-fraction.
            Z (ArrayLike): heavy-element mass-fraction.

        Raises:
            ValueError: input can at most be two-dimensional.

        Returns:
            NDArray: reduced equation of state output. The index of the
            individual quantities is defined in the __init__ method.
        """

        logT, logP = self.__check_PT(logT, logP)
        X, Y, Z = check_composition(X, Z)
        if logT.ndim > X.ndim:
            X = X * np.ones_like(logT)
            Y = Y * np.ones_like(logT)
            Z = Z * np.ones_like(logT)
        input_ndim = np.max([logT.ndim, X.ndim])

        if np.any(X > 0):
            res_x = self.__evaluate_x(logT, logP)
        else:
            res_x = self.__get_zeros(logT, logP, X, Z)
        if np.any(Y > 0):
            res_y = self.__evaluate_y(logT, logP)
        else:
            res_y = self.__get_zeros(logT, logP, X, Z)
        if np.any(Z > 0):
            res_z = self.__evaluate_z(logT, logP)
        else:
            res_z = self.__get_zeros(logT, logP, X, Z)

        logRho = self.__ideal_mixture(logT, logP, X, Y, Z)
        logRho_x = res_x[self.i_logRho]
        logRho_y = res_y[self.i_logRho]
        logRho_z = res_z[self.i_logRho]

        T = 10**logT
        P = 10**logP
        rho = 10**logRho
        rho_x = 10**logRho_x
        rho_y = 10**logRho_y
        rho_z = 10**logRho_z

        logS_x = res_x[self.i_logS]
        logS_y = res_y[self.i_logS]
        logS_z = res_z[self.i_logS]
        S_x = 10**logS_x
        S_y = 10**logS_y
        S_z = 10**logS_z
        S = X * S_x + Y * S_y + Z * S_z
        logS = np.log10(S)

        logU_x = res_x[self.i_logU]
        logU_y = res_y[self.i_logU]
        logU_z = res_z[self.i_logU]
        U = X * (10**logU_x) + Y * (10**logU_y) + Z * (10**logU_z)
        logU = np.log10(U)

        dlS_dlP_T_x = self.interpPT_dlS_dlP_T_x(logT, logP, **self.kwargs)
        dlS_dlT_P_x = self.interpPT_dlS_dlT_P_x(logT, logP, **self.kwargs)
        dlS_dlP_T_y = self.interpPT_dlS_dlP_T_y(logT, logP, **self.kwargs)
        dlS_dlT_P_y = self.interpPT_dlS_dlT_P_y(logT, logP, **self.kwargs)
        dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
        dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

        dlS_dlP_T = (
            X * S_x * dlS_dlP_T_x + Y * S_y * dlS_dlP_T_y + Z * S_z * dlS_dlP_T_z
        ) / S
        dlS_dlT_P = (
            X * S_x * dlS_dlT_P_x + Y * S_y * dlS_dlT_P_y + Z * S_z * dlS_dlT_P_z
        ) / S

        eps = 1e-4
        if input_ndim > 0:
            shape = (3, 3) + logT.shape
            fac = np.zeros(shape)
            iX = np.isclose(X, 0, atol=eps)
            iY = np.isclose(Y, 0, atol=eps)
            iZ = np.isclose(Z, 0, atol=eps)

            if np.any(iX) or np.any(iY) or np.any(iZ):
                i = X > eps
                fac[0, 0, i] = X[i] / rho_x[i] / res_x[self.i_chiRho, i]
                fac[0, 1, i] = -fac[0, 0, i] * res_x[self.i_chiT, i]
                fac[0, 2, i] = X[i] / res_x[self.i_mu, i]

                i = Y > eps
                fac[1, 0, i] = Y[i] / rho_y[i] / res_y[self.i_chiRho, i]
                fac[1, 1, i] = -fac[1, 0, i] * res_y[self.i_chiT, i]
                fac[1, 2, i] = Y[i] / res_y[self.i_mu, i]

                i = Z > eps
                fac[2, 0, i] = Z[i] / rho_z[i] / res_z[self.i_chiRho, i]
                fac[2, 1, i] = -fac[2, 0, i] * res_z[self.i_chiT, i]
                fac[2, 2, i] = Z[i] / res_z[self.i_mu, i]
            else:
                fac[0, 0] = X / rho_x / res_x[self.i_chiRho]
                fac[0, 1] = -fac[0, 0] * res_x[self.i_chiT]
                fac[0, 2] = X / res_x[self.i_mu]
                fac[1, 0] = Y / rho_y / res_y[self.i_chiRho]
                fac[1, 1] = -fac[1, 0] * res_x[self.i_chiT]
                fac[1, 2] = Y / res_y[self.i_mu]
                fac[2, 0] = Z / rho_z / res_z[self.i_chiRho]
                fac[2, 1] = -fac[2, 0] * res_z[self.i_chiT]
                fac[2, 2] = Z / res_z[self.i_mu]
        else:
            fac = np.zeros((3, 3))
            if X > eps:
                fac[0, 0] = X / rho_x / res_x[self.i_chiRho]
                fac[0, 1] = -fac[0, 0] * res_x[self.i_chiT]
                fac[0, 2] = X / res_x[self.i_mu]
            if Y > eps:
                fac[1, 0] = Y / rho_y / res_y[self.i_chiRho]
                fac[1, 1] = -fac[1, 0] * res_y[self.i_chiT]
                fac[1, 2] = Y / res_y[self.i_mu]
            if Z > eps:
                fac[2, 0] = Z / rho_z / res_z[self.i_chiRho]
                fac[2, 1] = -fac[2, 0] * res_z[self.i_chiT]
                fac[2, 2] = Z / res_z[self.i_mu]
        dlRho_dlP_T = rho * (fac[0, 0] + fac[1, 0] + fac[2, 0])
        dlRho_dlT_P = rho * (fac[0, 1] + fac[1, 1] + fac[2, 1])
        mu = 1 / (fac[0, 2] + fac[1, 2] + fac[2, 2])

        grad_ad = -dlS_dlP_T / dlS_dlT_P
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        if input_ndim > 0:
            grad_ad[grad_ad < 0] = 0
            chiRho[chiRho < 0] = 0
            chiT[chiT < 0] = 0
        else:
            grad_ad = np.max(grad_ad, 0)
            chiRho = np.max(chiRho, 0)
            chiT = np.max(chiT, 0)

        gamma1 = chiRho / (1 - chiT * grad_ad)
        gamma3 = 1 + gamma1 * grad_ad
        cp = S * dlS_dlT_P

        if input_ndim > 0:
            cv = np.zeros_like(logT)
            i = chiRho == 0
            if np.any(i):
                cv[i] = cp[i]
                i = ~i
                cv[i] = cp[i] * chiRho[i] / gamma1[i]
            else:
                cv = cp * chiRho / gamma1
            c_sound = np.zeros_like(logT)
            i = gamma1 >= 0
            c_sound[i] = np.sqrt(P[i] / rho[i] * gamma1[i])
        else:
            if chiRho == 0:
                cv = cp
            else:
                cv = cp * chiRho / gamma1
            if gamma1 >= 0:
                c_sound = np.sqrt(P / rho * gamma1)
            else:
                c_sound = 0

        # these are at constant density or temperature
        dS_dT = cv / T  # definition of specific heat
        dS_dRho = -(P / T / rho**2) * chiT  # maxwell relation
        dE_dRho = (P / rho**2) * (1 - chiT)

        # only hydrogen and helium contribute to free electrons
        lfe = np.log10(X * 10 ** res_x[self.i_lfe] + Y * 10 ** res_y[self.i_lfe])
        if np.any(np.isinf(lfe)):
            if input_ndim > 0:
                i = np.isinf(lfe)
                lfe[i] = -99
            else:
                lfe = -99
        eta = get_eta(logT, logRho, lfe)

        res = self.__get_zeros(logT, logP, X, Z)
        res[self.i_logT] = logT
        res[self.i_logRho] = logRho
        res[self.i_logP] = logP
        res[self.i_logS] = logS
        res[self.i_logU] = logU
        res[self.i_chiRho] = chiRho
        res[self.i_chiT] = chiT
        res[self.i_grad_ad] = grad_ad
        res[self.i_cp] = cp
        res[self.i_cv] = cv
        res[self.i_gamma1] = gamma1
        res[self.i_gamma3] = gamma3
        res[self.i_dS_dT] = dS_dT
        res[self.i_dS_dRho] = dS_dRho
        res[self.i_dE_dRho] = dE_dRho
        res[self.i_mu] = mu
        res[self.i_eta] = eta
        res[self.i_lfe] = lfe
        res[self.i_csound] = c_sound
        return res
