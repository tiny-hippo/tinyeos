import os
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Tuple
from pathlib import Path
from scipy.optimize import root_scalar
from tinyeos.tinypteos import TinyPT
from tinyeos.interpolantsbuilder import InterpolantsBuilder
from tinyeos.support import (
    A_H,
    A_He,
    m_u,
    k_b,
    get_eta,
    ideal_mixing_law,
    check_composition,
    get_h_he_number_fractions,
)
from tinyeos.definitions import (
    logP_max,
    logP_min,
    logT_max,
    logT_min,
    logRho_max,
    logRho_min,
    num_vals,
    i_logT,
    i_logRho,
    i_logP,
    i_logS,
    i_logU,
    i_chiRho,
    i_chiT,
    i_grad_ad,
    i_cp,
    i_cv,
    i_gamma1,
    i_gamma3,
    i_dS_dT,
    i_dS_dRho,
    i_dE_dRho,
    i_mu,
    i_eta,
    i_lfe,
    i_csound,
    eps1,
    tiny_val,
    tiny_logRho,
    heavy_elements,
    atomic_masses,
    ionic_charges,
)


class TinyDT(InterpolantsBuilder):
    """Temperature-density equation of state for a mixture of hydrogen,
    helium and a heavy element. Units are cgs everywhere.

    Equations of state implemented:
        Hydrogen-Helium:
            CMS (Chabrier et al. 2019),
            SCvH (Saumon et al. 1995).

        Heavy element:
            H2O (QEOS, More et al. 1988),
            SiO2 (QEOS, More et al. 1988),
            Fe (QEOS, More et al. 1998),
            CO (QEOS, Podolak et al. 2022),
            ideal mixture of H2O and SiO2 (QEOS, More et al. 1988).
    """

    def __init__(
        self,
        which_heavy: str = "h2o",
        which_hhe: str = "cms",
        include_hhe_interactions: bool = False,
        use_smoothed_xy_tables: bool = False,
        use_smoothed_z_tables: bool = False,
        build_interpolants: bool = False,
    ) -> None:
        """__init__ method. Defines parameters and either loads or
        builds the interpolants.

        Args:
            which_heavy (str, optional): which heavy-element equation of state
                to use. Defaults to "h2o". Options are "h2o", "sio2",
                "mixture", "fe" or "co".
            which_hhe (str, optional): which hydrogen-helium equation of state
                to use. Defaults to "cms". Options are "cms" or "scvh".
            include_hhe_interactions (bool, optional): wether to include
                hydrogen-helium interactions. Defaults to False.
            use_smoothed_xy_tables (bool, optional): whether to use smoothed
                hydrogen and helium tables. Defaults to False.
            use_smoothed_z_tables (bool, optional): whether to use smoothed
                heavy-element tables. Defaults to False.
            build_interpolants (bool, optional): whether to build interpolants.
                Defaults to False.

        Raises:
            NotImplementedError: raised if which_heavy or which_hhe choices
                are unavailable.
        """

        self.tpt = TinyPT(
            which_heavy=which_heavy,
            which_hhe=which_hhe,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
            build_interpolants=build_interpolants,
        )

        if build_interpolants:
            super().__init__()

        self.logRho_max = logRho_max
        self.logRho_min = logRho_min
        self.logP_min = logP_min
        self.logP_max = logP_max
        self.logT_max = logT_max
        self.logT_min = logT_min

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
        if which_heavy not in heavy_elements:
            raise NotImplementedError("invalid option for which_heavy")
        if which_hhe not in ["cms", "scvh"]:
            raise NotImplementedError("invalid option for which_hhe")
        # to-do: scvh is currently giving inconsistent results between
        # TinyPT and TinyDT; needs to be fixed before allowing it again
        if which_hhe == "scvh":
            raise NotImplementedError("scvh is currently disabled")
        self.include_hhe_interactions = include_hhe_interactions
        if include_hhe_interactions and which_hhe == "scvh":
            raise NotImplementedError("can't include H-He interactions with scvh")

        # heavy-element atomic mass and ionic charge
        self.heavy_element = which_heavy
        self.A = atomic_masses[which_heavy]
        self.z = ionic_charges[which_heavy]

        # if use_smoothed_xy_tables:
        #     which_hhe = which_hhe + "_smoothed"
        if use_smoothed_z_tables:
            which_heavy = which_heavy + "_smoothed"
        self.interpPT_x = self.__load_interp("interpPT_x_" + which_hhe + ".npy")
        self.interpPT_y = self.__load_interp("interpPT_y_" + which_hhe + ".npy")
        self.interpPT_z = self.__load_interp("interpPT_z_" + which_heavy + ".npy")

        self.interpDT_x = self.__load_interp("interpDT_x_" + which_hhe + ".npy")
        self.interpDT_y = self.__load_interp("interpDT_y_" + which_hhe + ".npy")
        self.interpDT_z = self.__load_interp("interpDT_z_" + which_heavy + ".npy")

        if self.include_hhe_interactions:
            self.interpPT_x_eff = self.__load_interp(
                "interpPT_x_eff_" + which_hhe + ".npy"
            )
            self.interpDT_x_eff = self.__load_interp(
                "interpDT_x_eff_" + which_hhe + ".npy"
            )

        self.interpDT_logP_x = self.interpDT_x[0]
        self.interpDT_logS_x = self.interpDT_x[1]
        self.interpDT_logU_x = self.interpDT_x[2]
        self.interpDT_dlRho_dlT_P_x = self.interpDT_x[3]
        self.interpDT_dlRho_dlP_T_x = self.interpDT_x[4]
        self.interpDT_dlS_dlT_P_x = self.interpDT_x[5]
        self.interpDT_dlS_dlP_T_x = self.interpDT_x[6]
        self.interpDT_grad_ad_x = self.interpDT_x[7]
        self.interpDT_lfe_x = self.interpDT_x[8]
        self.interpDT_mu_x = self.interpDT_x[9]

        if self.include_hhe_interactions:
            self.interpDT_logP_x_eff = self.interpDT_x_eff[0]
            self.interpDT_logS_x_eff = self.interpDT_x_eff[1]
            self.interpDT_logU_x_eff = self.interpDT_x_eff[2]
            self.interpDT_dlRho_dlT_P_x_eff = self.interpDT_x_eff[3]
            self.interpDT_dlRho_dlP_T_x_eff = self.interpDT_x_eff[4]
            self.interpDT_dlS_dlT_P_x_eff = self.interpDT_x_eff[5]
            self.interpDT_dlS_dlP_T_x_eff = self.interpDT_x_eff[6]
            self.interpDT_grad_ad_x_eff = self.interpDT_x_eff[7]
            self.interpDT_lfe_x_eff = self.interpDT_x_eff[8]
            self.interpDT_mu_x_eff = self.interpDT_x_eff[9]

        self.interpDT_logP_y = self.interpDT_y[0]
        self.interpDT_logS_y = self.interpDT_y[1]
        self.interpDT_logU_y = self.interpDT_y[2]
        self.interpDT_dlRho_dlT_P_y = self.interpDT_y[3]
        self.interpDT_dlRho_dlP_T_y = self.interpDT_y[4]
        self.interpDT_dlS_dlT_P_y = self.interpDT_y[5]
        self.interpDT_dlS_dlP_T_y = self.interpDT_y[6]
        self.interpDT_grad_ad_y = self.interpDT_y[7]
        self.interpDT_lfe_y = self.interpDT_y[8]
        self.interpDT_mu_y = self.interpDT_y[9]

        self.interpDT_logP_z = self.interpDT_z[0]
        self.interpDT_logS_z = self.interpDT_z[1]
        self.interpDT_logU_z = self.interpDT_z[2]
        if which_heavy == "aqua":
            self.interpDT_grad_ad_z = self.interpDT_z[3]

        self.interpPT_logRho_x = self.interpPT_x[0]
        if self.include_hhe_interactions:
            self.interpPT_logRho_x_eff = self.interpPT_x_eff[0]
        self.interpPT_logRho_y = self.interpPT_y[0]
        self.interpPT_logRho_z = self.interpPT_z[0]
        self.interpPT_logS_z = self.interpPT_z[1]
        self.interpPT_logU_z = self.interpPT_z[2]

    def __call__(self, logT: float, logRho: float, X: float, Z: float) -> NDArray:
        """__call__ method acting as convenience wrapper for the evaluate
        method. Calculates the equation of state output for the mixture.

        Args:
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            X (float): hydrogen mass-fraction.
            Z (float): heavy-element mass-fraction.

        Returns:
            NDArray: Equation of state output. The index of the individual
                quantities is defined in the __init__ method. If the root finding
                algorithm failed, the output will be filled with negative ones.
        """
        return self.evaluate(logT, logRho, X, Z)

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
            raise FileNotFoundError("missing interpolant cache " + src)
        return np.load(src, allow_pickle=True)

    def __check_DT(self, logT: ArrayLike, logRho: ArrayLike) -> None:
        """Makes sure that input temperature and density
        are within equation of state limits.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.

        Raises:
            ValueError: logT and logRho must have equal shape
                and all values must be within the equation of
                state limits.

        Returns:
            Tuple[NDArray, NDArray]: (logT, logRho) as arrays.
        """

        if not isinstance(logT, np.ndarray):
            logT = np.array(logT, dtype=np.float64)
        if not isinstance(logRho_max, np.ndarray):
            logRho = np.array(logRho, dtype=np.float64)

        if not logT.shape == logRho.shape:
            msg = "logT and logRho must have equal shape"
            raise ValueError(msg)
        if np.any(logT < self.logT_min) or np.any(logT > self.logT_max):
            msg = "logT out of bounds"
            raise ValueError(msg)
        elif np.any(logRho < self.logRho_min) or np.any(logRho > self.logRho_max):
            msg = "logRho out of bounds"
            raise ValueError(msg)
        else:
            return (logT, logRho)

    def __get_zeros(
        self,
        logT: NDArray,
        logRho: NDArray,
        X: NDArray = np.array(0),
        Z: NDArray = np.array(0),
    ) -> NDArray:
        """Helper function to return a result array of the appropriate shape

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            X (ArrayLike): hydrogen mass-fraction.
            Z (ArrayLike): heavy-element mass fraction.

        Raises:
            ValueError: input can at most be two-dimensional.

        Returns:
            NDArray
        """
        max_ndim = np.max([logT.ndim, logRho.ndim, X.ndim, Z.ndim])
        if max_ndim > 0:
            shape = (self.num_vals,) + logT.shape
            return np.zeros(shape)
        elif max_ndim == 0:
            return np.zeros(self.num_vals)
        else:
            msg = "unsupported input shape"
            raise ValueError(msg)

    def __get_mixing_entropy(self, Y: ArrayLike) -> ArrayLike:
        """Calculates the ideal mixing entropy of the H-He
        partial mixture with free-electron entropy neglected;
        see eq. 11 of Chabrier et al. (2019)

        Args:
            Y (ArrayLike): helium mass-fraction.

        Returns:
            ArrayLike: mixing entropy.
        """
        S_mix = np.zeros(Y.shape)
        if not self.include_hhe_interactions:
            x_H, x_He = get_h_he_number_fractions(Y)
            if self.input_ndim > 0:
                if not np.all(self.Z_close):
                    iZ = ~self.Z_close
                    x_H = x_H[iZ]
                    x_He = x_He[iZ]
                    mean_A = x_H * A_H + x_He * A_He
                    S_mix[iZ] = (
                        -k_b
                        * (x_H * np.log(x_H) + x_He * np.log(x_He))
                        / (mean_A * m_u)
                    )
            else:
                if not self.Z_close:
                    mean_A = x_H * A_H + x_He * A_He
                    S_mix = (
                        -k_b
                        * (x_H * np.log(x_H) + x_He * np.log(x_He))
                        / (mean_A * m_u)
                    )
        return S_mix

    def __ideal_mixture(
        self,
        logT: float,
        logRho: float,
        X: float,
        Y: float,
        Z: float,
    ) -> Tuple:
        """Calculates the individual densities of the elements in the mixture.

        Args:
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            X (float): hydrogen mass-fraction
            Y (float): helium mass-fraction.
            Z (float): heavy-element mass-fraction.

        Returns:
            Tuple: tuple consisting of convergence information, individual
                densities and gas pressure.
        """
        if np.isclose(Z, 1, atol=eps1):
            conv = True
            logP = self.interpDT_logP_z(logT, logRho, **self.kwargs)
            logRho_x = tiny_logRho
            logRho_y = tiny_logRho
            logRho_z = logRho
        elif np.isclose(X, 1, atol=eps1):
            conv = True
            logP = self.interpDT_logP_x(logT, logRho, **self.kwargs)
            logRho_x = logRho
            logRho_y = tiny_logRho
            logRho_z = tiny_logRho
        elif np.isclose(Y, 1, atol=eps1):
            conv = True
            logP = self.interpDT_logP_y(logT, logRho, **self.kwargs)
            logRho_x = tiny_logRho
            logRho_y = logRho
            logRho_z = tiny_logRho
        else:
            logP0 = self.logP_min
            logP1 = self.logP_max
            f1 = self.__ideal_mixing_law_wrapper(logP0, logT, logRho, X, Y, Z)
            f2 = self.__ideal_mixing_law_wrapper(logP1, logT, logRho, X, Y, Z)
            if np.sign(f1) == np.sign(f2):
                conv = False
                logRho_x = np.nan
                logRho_y = np.nan
                logRho_z = np.nan
                logP = np.nan
            else:
                sol = root_scalar(
                    self.__ideal_mixing_law_wrapper,
                    args=(logT, logRho, X, Y, Z),
                    method="brentq",
                    bracket=[logP0, logP1],
                )
                conv = sol.converged
                logP = sol.root
                logRho_x = self.interpPT_logRho_x(logT, logP, **self.kwargs)
                logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)
                logRho_z = self.interpPT_logRho_z(logT, logP, **self.kwargs)

            # dlogP = 0.01
            # logPs = np.arange(self.logP_min, self.logP_max, dlogP)
            # logTs = logT * np.ones_like(logPs)
            # logRhos = self.tpt.evaluate(logTs, logPs, X, Z)
            # i = np.where(np.isclose(logRho, logRhos, rtol=1e-3))
            # print(i)
        return (conv, logRho_x, logRho_y, logRho_z, logP)

    def __ideal_mixing_law_wrapper(
        self,
        logP: float,
        logT: float,
        logRho: float,
        X: float,
        Y: float,
        Z: float,
    ) -> float:
        if X > tiny_val:
            logRho_x = self.interpPT_logRho_x(logT, logP, **self.kwargs)
        else:
            logRho_x = tiny_logRho
        if Y > tiny_val:
            logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)
        else:
            logRho_y = tiny_logRho
        if Z > tiny_val:
            logRho_z = self.interpPT_logRho_z(logT, logP, **self.kwargs)
        else:
            logRho_z = tiny_logRho

        rho_x = 10**logRho_x
        rho_y = 10**logRho_y
        rho_z = 10**logRho_z
        log_iml = np.log10(ideal_mixing_law(rho_x, rho_y, rho_z, X, Y, Z))
        val = log_iml + logRho
        return val

    def __evaluate_x(self, logT: ArrayLike, logRho: ArrayLike) -> NDArray:
        """Calculates equation of state output for hydrogen.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.

        Returns:
            NDArray: equation of state output.
        """

        logP = self.interpDT_logP_x(logT, logRho, **self.kwargs)
        logS = self.interpDT_logS_x(logT, logRho, **self.kwargs)
        logU = self.interpDT_logU_x(logT, logRho, **self.kwargs)

        dlRho_dlP_T = self.interpDT_dlRho_dlP_T_x(logT, logRho, **self.kwargs)
        dlRho_dlT_P = self.interpDT_dlRho_dlT_P_x(logT, logRho, **self.kwargs)

        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        grad_ad = self.interpDT_grad_ad_x(logT, logRho, **self.kwargs)
        lfe = self.interpDT_lfe_x(logT, logRho, **self.kwargs)
        mu = self.interpDT_mu_x(logT, logRho, **self.kwargs)

        res_x = self.__get_zeros(logT, logRho)
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

    def __evaluate_x_eff(self, logT: ArrayLike, logRho: ArrayLike) -> NDArray:
        """Calculates equation of state output for effective hydrogen.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.

        Returns:
            NDArray: equation of state output.
        """

        logP = self.interpDT_logP_x_eff(logT, logRho, **self.kwargs)
        logS = self.interpDT_logS_x_eff(logT, logRho, **self.kwargs)
        logU = self.interpDT_logU_x_eff(logT, logRho, **self.kwargs)

        dlRho_dlP_T = self.interpDT_dlRho_dlP_T_x_eff(logT, logRho, **self.kwargs)
        dlRho_dlT_P = self.interpDT_dlRho_dlT_P_x_eff(logT, logRho, **self.kwargs)

        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        grad_ad = self.interpDT_grad_ad_x_eff(logT, logRho, **self.kwargs)
        lfe = self.interpDT_lfe_x_eff(logT, logRho, **self.kwargs)
        mu = self.interpDT_mu_x_eff(logT, logRho, **self.kwargs)

        res_x_eff = self.__get_zeros(logT, logRho)
        res_x_eff[self.i_logT] = logT
        res_x_eff[self.i_logRho] = logRho
        res_x_eff[self.i_logP] = logP
        res_x_eff[self.i_logS] = logS
        res_x_eff[self.i_logU] = logU
        res_x_eff[self.i_chiRho] = chiRho
        res_x_eff[self.i_chiT] = chiT
        res_x_eff[self.i_grad_ad] = grad_ad
        res_x_eff[self.i_mu] = mu
        res_x_eff[self.i_lfe] = lfe
        return res_x_eff

    def __evaluate_y(self, logT: ArrayLike, logRho: ArrayLike) -> NDArray:
        """Calculates equation of state output for hydrogen.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.

        Returns:
            NDArray: equation of state output.
        """

        logP = self.interpDT_logP_y(logT, logRho, **self.kwargs)
        logS = self.interpDT_logS_y(logT, logRho, **self.kwargs)
        logU = self.interpDT_logU_y(logT, logRho, **self.kwargs)

        dlRho_dlP_T = self.interpDT_dlRho_dlP_T_y(logT, logRho, **self.kwargs)
        dlRho_dlT_P = self.interpDT_dlRho_dlT_P_y(logT, logRho, **self.kwargs)

        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        grad_ad = self.interpDT_grad_ad_y(logT, logRho, **self.kwargs)
        lfe = self.interpDT_lfe_y(logT, logRho, **self.kwargs)
        mu = self.interpDT_mu_y(logT, logRho, **self.kwargs)

        res_y = self.__get_zeros(logT, logRho)
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

    def __evaluate_z(self, logT: ArrayLike, logRho: ArrayLike) -> NDArray:
        """Calculates equation of state output for the heavy element.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.

        Returns:
            NDArray: equation of state output.
        """

        logP = self.interpDT_logP_z(logT, logRho, **self.kwargs)
        # logS = self.interpDT_logS_z(logT, logRho, **self.kwargs)
        # use (logT, logP) for logS to be consistent
        # with the upcoming derivatives
        logS = self.interpPT_logS_z(logT, logP, **self.kwargs)
        logU = self.interpDT_logU_z(logT, logRho, **self.kwargs)

        # new method with (logT, logP)
        dlRho_dlP_T = self.interpPT_logRho_z(logT, logP, dy=1, **self.kwargs)
        dlRho_dlT_P = self.interpPT_logRho_z(logT, logP, dx=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        if self.heavy_element == "aqua":
            grad_ad = self.interpDT_grad_ad_z(logT, logRho, **self.kwargs)
        else:
            dlS_dlP_T = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
            dlS_dlT_P = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)
            grad_ad = -dlS_dlP_T / dlS_dlT_P

        # old method with (logT, logRho)
        # chiRho = self.interpDT_logP_z(logT, logRho, dy=1, **self.kwargs)
        # chiT = self.interpDT_logP_z(logT, logRho, dx=1, **self.kwargs)
        # dlS_dlT_rho = self.interpDT_logS_z(logT, logRho, dx=1, **self.kwargs)
        # dlS_dlRho_T = self.interpDT_logS_z(logT, logRho, dy=1, **self.kwargs)
        # dlS_dlRho_P = dlS_dlRho_T + dlS_dlT_rho * (- chiRho / chiT)
        # dlS_dlP_rho = dlS_dlT_rho / chiT
        # gamma1 = - dlS_dlRho_P / dlS_dlP_rho
        # grad_ad = (1 - chiRho / gamma1) / chiT
        # alternatively:
        # grad_ad = 1 / (chiT - dlS_dlT_rho * chiRho / dlS_dlRho_T)

        res_z = self.__get_zeros(logT, logRho)
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

    def evaluate(
        self,
        logT: ArrayLike,
        logRho: ArrayLike,
        X: ArrayLike,
        Z: ArrayLike,
        verbose: bool = False,
    ) -> NDArray:
        """Calculates the equation of state output for the mixture. For now,
        this only supports scalar inputs.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            X (ArrayLike): hydrogen mass-fraction.
            Z (ArrayLike): heavy-element mass-fraction.

        Raises:
            ValueError: input can at most be two-dimensional.

        Returns:
            NDArray: reduced equation of state output. The indices of the
                individual quantities are defined in the __init__ method.
        """

        logT, logRho = self.__check_DT(logT, logRho)
        X, Y, Z = check_composition(X, Z)
        if logT.ndim > X.ndim:
            X = X * np.ones_like(logT)
            Y = Y * np.ones_like(logT)
            Z = Z * np.ones_like(logT)
        elif logT.ndim < X.ndim:
            logT = logT * np.ones_like(X)
            logRho = logRho * np.ones_like(X)
        self.input_ndim = np.max([logT.ndim, X.ndim])

        self.X_close = np.isclose(X, 1, atol=eps1)
        self.Y_close = np.isclose(X, 1, atol=eps1)
        self.Z_close = np.isclose(Z, 1, atol=eps1)

        res = self.__get_zeros(logT, logRho, X, Z)
        iml = self.__ideal_mixture(logT, logRho, X, Y, Z)
        if not iml[0]:
            if verbose:
                msg = "evaluate failed in density root find"
                print(msg)
            res[self.i_logT] = logT
            res[self.i_logRho] = logRho
            res[self.i_logP :] = np.nan
            return res

        logRho_x = np.array(iml[1], dtype=np.float64)
        logRho_y = np.array(iml[2], dtype=np.float64)
        logRho_z = np.array(iml[3], dtype=np.float64)
        logP = iml[4]

        if np.all(self.X_close) or not self.include_hhe_interactions:
            res_x = self.__evaluate_x(logT, logRho_x)
        elif np.any(self.X_close) and self.include_hhe_interactions:
            i_x_eff = X < 1
            i_x = ~i_x_eff
            logT_x_normal = logT[i_x]
            logRho_x_normal = logRho_x[i_x]
            logT_x_eff = logT[i_x_eff]
            logRho_x_eff = logRho_x[i_x_eff]
            res_x = self.__get_zeros(logT, logRho, X, Z)
            res_x[:, i_x] = self.__evaluate_x(logT_x_normal, logRho_x_normal)
            res_x[:, i_x_eff] = self.__evaluate_x_eff(logT_x_eff, logRho_x_eff)
        else:
            res_x = self.__evaluate_x_eff(logT, logRho_x)
        res_y = self.__evaluate_y(logT, logRho_y)
        res_z = self.__evaluate_z(logT, logRho_z)

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
        S = S + self.__get_mixing_entropy(Y)
        logS = np.log10(S)

        logU_x = res_x[self.i_logU]
        logU_y = res_y[self.i_logU]
        logU_z = res_z[self.i_logU]
        U = X * (10**logU_x) + Y * (10**logU_y) + Z * (10**logU_z)
        logU = np.log10(U)

        if np.all(self.X_close) or not self.include_hhe_interactions:
            dlS_dlP_T_x = self.interpDT_dlS_dlP_T_x(logT, logRho_x, **self.kwargs)
            dlS_dlT_P_x = self.interpDT_dlS_dlT_P_x(logT, logRho_x, **self.kwargs)
        elif np.any(self.X_close) and self.include_hhe_interactions:
            dlS_dlP_T_x = np.zeros_like(logS)
            dlS_dlP_T_x[i_x] = self.interpDT_dlS_dlP_T_x(
                logT_x_normal, logRho_x_normal, **self.kwargs
            )
            dlS_dlP_T_x[i_x_eff] = self.interpDT_dlS_dlP_T_x(
                logT_x_eff, logRho_x_eff, **self.kwargs
            )

            dlS_dlT_P_x = np.zeros_like(logS)
            dlS_dlT_P_x[i_x] = self.interpDT_dlS_dlT_P_x(
                logT_x_normal, logRho_x_normal, **self.kwargs
            )
            dlS_dlT_P_x[i_x_eff] = self.interpDT_dlS_dlT_P_x(
                logT_x_eff, logRho_x_eff, **self.kwargs
            )
        else:
            dlS_dlP_T_x = self.interpDT_dlS_dlP_T_x_eff(logT, logRho_x, **self.kwargs)
            dlS_dlT_P_x = self.interpDT_dlS_dlT_P_x_eff(logT, logRho_x, **self.kwargs)

        dlS_dlP_T_y = self.interpDT_dlS_dlP_T_y(logT, logRho_y, **self.kwargs)
        dlS_dlT_P_y = self.interpDT_dlS_dlT_P_y(logT, logRho_y, **self.kwargs)
        dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
        dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

        dlS_dlP_T = (
            X * S_x * dlS_dlP_T_x + Y * S_y * dlS_dlP_T_y + Z * S_z * dlS_dlP_T_z
        ) / S
        dlS_dlT_P = (
            X * S_x * dlS_dlT_P_x + Y * S_y * dlS_dlT_P_y + Z * S_z * dlS_dlT_P_z
        ) / S

        if self.input_ndim > 0:
            shape = (3, 3) + logT.shape
            fac = np.zeros(shape)
            iX = np.isclose(X, tiny_val, atol=eps1)
            iY = np.isclose(Y, tiny_val, atol=eps1)
            iZ = np.isclose(Z, tiny_val, atol=eps1)
            if np.any(iX) or np.any(iY) or np.any(iZ):
                i = X > tiny_val
                fac[0, 0, i] = X[i] / rho_x[i] / res_x[self.i_chiRho, i]
                fac[0, 1, i] = -fac[0, 0, i] * res_x[self.i_chiT, i]
                fac[0, 2, i] = X[i] / res_x[self.i_mu, i]

                i = Y > tiny_val
                fac[1, 0, i] = Y[i] / rho_y[i] / res_y[self.i_chiRho, i]
                fac[1, 1, i] = -fac[1, 0, i] * res_y[self.i_chiT, i]
                fac[1, 2, i] = Y[i] / res_y[self.i_mu, i]

                i = Z > tiny_val
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
            if X > tiny_val:
                fac[0, 0] = X / rho_x / res_x[self.i_chiRho]
                fac[0, 1] = -fac[0, 0] * res_x[self.i_chiT]
                fac[0, 2] = X / res_x[self.i_mu]
            if Y > tiny_val:
                fac[1, 0] = Y / rho_y / res_y[self.i_chiRho]
                fac[1, 1] = -fac[1, 0] * res_y[self.i_chiT]
                fac[1, 2] = Y / res_y[self.i_mu]
            if Z > tiny_val:
                fac[2, 0] = Z / rho_z / res_z[self.i_chiRho]
                fac[2, 1] = -fac[2, 0] * res_z[self.i_chiT]
                fac[2, 2] = Z / res_z[self.i_mu]
        dlRho_dlP_T = rho * (fac[0, 0] + fac[1, 0] + fac[2, 0])
        dlRho_dlT_P = rho * (fac[0, 1] + fac[1, 1] + fac[2, 1])
        mu = 1 / (fac[0, 2] + fac[1, 2] + fac[2, 2])

        grad_ad = -dlS_dlP_T / dlS_dlT_P
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        if self.input_ndim > 0:
            grad_ad[np.isnan(grad_ad)] = tiny_val
            grad_ad[grad_ad < 0.1] = 0.1
            grad_ad[grad_ad > 0.5] = 0.5
            chiT[chiT <= tiny_val] = tiny_val
            i = chiRho <= tiny_val
            chiRho[i] = tiny_val

            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            cp = S * dlS_dlT_P

            cv = np.zeros_like(logT)
            if np.any(i):
                cv[i] = cp[i]
                i = ~i
                cv[i] = cp[i] * chiRho[i] / gamma1[i]
                # Chabrier et al. (2019) eq. 5:
                # cv[i] = cp[i] - (P[i] * chiT[i]**2) / (rho[i] * T[i] * chiRho)
            else:
                cv = cp * chiRho / gamma1
                # cv = cp - (P * chiT**2) / (rho * T * chiRho)

            c_sound = tiny_val * np.ones_like(logT)
            i = gamma1 >= tiny_val
            c_sound[i] = np.sqrt(P[i] / rho[i] * gamma1[i])
        else:
            if np.isnan(grad_ad):
                grad_ad = tiny_val
            grad_ad = np.min([np.max([grad_ad, 0.1]), 0.5])
            chiRho = np.max([chiRho, tiny_val])
            chiT = np.max([chiT, tiny_val])

            gamma1 = chiRho / (1 - chiT * grad_ad)
            gamma3 = 1 + gamma1 * grad_ad
            cp = S * dlS_dlT_P

            if np.isclose(chiRho, tiny_val, atol=eps1):
                cv = cp
            else:
                cv = cp * chiRho / gamma1
            if gamma1 >= tiny_val:
                c_sound = np.sqrt(P / rho * gamma1)
            else:
                c_sound = tiny_val

        # these are at constant density or temperature
        dS_dT = cv / T  # definition of specific heat
        dS_dRho = -(P / T / rho**2) * chiT  # maxwell relation
        dE_dRho = (P / rho**2) * (1 - chiT)

        # only hydrogen and helium contribute to free electrons
        lfe = np.log10(X * 10 ** res_x[self.i_lfe] + Y * 10 ** res_y[self.i_lfe])
        if self.input_ndim > 0:
            lfe[lfe < -99] = -99
        else:
            lfe = np.max([lfe, -99])
        eta = get_eta(logT, logRho, lfe)

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
