import os
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyeos.definitions import heavy_elements
from tinyeos.interpolantsbuilder import InterpolantsBuilder
from tinyeos.support import (
    check_composition,
    get_eta,
    get_mixing_entropy,
    get_zeros,
    ideal_mixing_law,
    set_eos_params,
)


class TinyPT(InterpolantsBuilder):
    """
    Equation of state for mixtures of hydrogen, helium, and heavy elements.
    This class provides a high-level interface for calculating the equation of state
    properties for mixtures of hydrogen, helium, and heavy elements.

    Parameters
    ----------
    which_hhe : str, optional
        Hydrogen-helium tables to use. Options: "cms", "scvh", "scvh_extended".
        Default: "cms".
    which_heavy : str, optional
        Heavy-element tables to use. Options: "h2o", "sesame_h2o", "aqua", "sio2",
        "fe", "co", and "mixture". Default: "h2o".
    Z1 : float, optional
        Mass fraction of H2O in the heavy-element mixture. Default: 0.5.
    Z2 : float, optional
        Mass fraction of SiO2 in the heavy-element mixture. Default: 0.5.
    Z3 : float, optional
        Mass fraction of Fe in the heavy-element mixture. Default: 0.0.
    include_hhe_interactions : bool, optional
        Whether to include hydrogen-helium interactions. Default: False.
    use_smoothed_xy_tables : bool, optional
        Use smoothed hydrogen and helium tables. Default: False.
    use_smoothed_z_tables : bool, optional
        Use smoothed heavy-element tables. Default: False.
    build_interpolants : bool, optional
        Build interpolants for EOS tables. Default: False.

    Raises
    ------
    NotImplementedError
        Raised if which_heavy or which_hhe are invalid choices.

    Methods
    -------
    __call__(logT, logP, X, Z)
        Convenience wrapper for "evaluate".
    evaluate(logT, logRho, X, Z)
        Calculates equation of state output for the mixture.
        Returns an NDArray with equation of state quantities.
    __prepare(logT, logP, X, Z)
        Prepares and validates input arrays for evaluation.
    __load_interp(filename)
        Loads an interpolant from disk.
    __check_pt(logT, logP)
        Validates that logT and logP are within limits.
    __ideal_mixture(logT, logP, X, Y, Z)
        Computes the total density using the ideal mixing law.
    __evaluate_x(logT, logP)
        Computes equation of state output for hydrogen.
    __evaluate_x_eff(logT, logP)
        Computes equation of state output for
        effective hydrogen (with H-He interactions).
    __evaluate_y(logT, logP)
        Computes equation of state output for helium.
    __evaluate_z(logT, logP)
        Computes equation of state output for the heavy element.
    """

    def __init__(
        self,
        which_hhe: str = "cms",
        which_heavy: str = "h2o",
        Z1: float = 0.5,
        Z2: float = 0.5,
        Z3: float = 0.0,
        include_hhe_interactions: bool = False,
        use_smoothed_xy_tables: bool = False,
        use_smoothed_z_tables: bool = False,
        build_interpolants: bool = False,
    ) -> None:
        """__init__ method. Defines parameters and either loads or
        builds the interpolants.

        Args:
            which_hhe (str, optional): hydrogen-helium equation of state
                to use. Defaults to "cms". Options are "cms", "scvh" or "scvh_extended".
            which_heavy (str, optional): heavy-element equation of state
                to use. Defaults to "h2o". Options are "h2o", "aqua", "sio2",
                "mixture", "fe" or "co".
            Z1 : float, optional
                Mass fraction of H2O in the heavy-element mixture. Default: 0.5.
            Z2 : float, optional
                Mass fraction of SiO2 in the heavy-element mixture. Default: 0.5.
            Z3 : float, optional
                Mass fraction of Fe in the heavy-element mixture. Default: 0.0.
            include_hhe_interactions (bool, optional): include
                hydrogen-helium interactions. Defaults to False.
            use_smoothed_xy_tables (bool, optional): use smoothed
                hydrogen and helium tables. Defaults to False.
            use_smoothed_z_tables (bool, optional): use smoothed
                heavy-element tables. Defaults to False.
            build_interpolants (bool, optional): build interpolants.
                Defaults to False.

        Raises:
            NotImplementedError: raised if which_heavy or which_hhe choices
                are unavailable.
        """

        self.kwargs = {"grid": False}
        self.cache_path = Path(__file__).parent / "data/eos/interpolants"
        if which_heavy not in heavy_elements:
            raise NotImplementedError("invalid option for which_heavy")
        if which_hhe not in ["cms", "scvh", "scvh_extended"]:
            raise NotImplementedError("invalid option for which_hhe")
        self.include_hhe_interactions = include_hhe_interactions
        if include_hhe_interactions and which_hhe == "scvh":
            raise NotImplementedError("can't include H-He interactions with scvh")

        if build_interpolants:
            super().__init__()

        set_eos_params(
            eos=self, which_hhe=which_hhe, which_heavy=which_heavy, Z1=Z1, Z2=Z2, Z3=Z3
        )

        # if use_smoothed_xy_tables:
        #     which_hhe = which_hhe + "_smoothed"
        if use_smoothed_z_tables:
            self.which_heavy = self.which_heavy + "_smoothed"
        self.interp_pt_x = self.__load_interp("interp_pt_x_" + which_hhe + ".npy")
        self.interp_pt_y = self.__load_interp("interp_pt_y_" + which_hhe + ".npy")
        try:
            self.interp_pt_z = self.__load_interp(
                "interp_pt_z_" + self.which_heavy + ".npy"
            )
        except FileNotFoundError:
            super().build_z_mixture_interpolants(Z1=Z1, Z2=Z2, Z3=Z3)
            self.interp_pt_z = self.__load_interp(
                "interp_pt_z_" + self.which_heavy + ".npy"
            )
        self.interp_dt_z = self.__load_interp(
            "interp_dt_z_" + self.which_heavy + ".npy"
        )

        if self.include_hhe_interactions:
            self.interp_pt_x_eff = self.__load_interp(
                "interp_pt_x_eff_" + which_hhe + ".npy"
            )
            self.interp_pt_xy = self.__load_interp("interp_pt_xy_int.npy")

        self.interp_dt_z = self.__load_interp(
            "interp_dt_z_" + self.which_heavy + ".npy"
        )

        self.interp_pt_logRho_x = self.interp_pt_x[0]
        self.interp_pt_logS_x = self.interp_pt_x[1]
        self.interp_pt_logU_x = self.interp_pt_x[2]
        self.interp_pt_lfe_x = self.interp_pt_x[3]
        self.interp_pt_mu_x = self.interp_pt_x[4]

        if self.include_hhe_interactions:
            self.interp_pt_logRho_x_eff = self.interp_pt_x_eff[0]
            self.interp_pt_logS_x_eff = self.interp_pt_x_eff[1]
            self.interp_pt_logU_x_eff = self.interp_pt_x_eff[2]
            self.interp_pt_lfe_x_eff = self.interp_pt_x_eff[3]
            self.interp_pt_mu_x_eff = self.interp_pt_x_eff[4]
            self.interp_pt_V_mix_xy = self.interp_pt_xy[0]
            self.interp_pt_S_mix_xy = self.interp_pt_xy[1]

        self.interp_pt_logRho_y = self.interp_pt_y[0]
        self.interp_pt_logS_y = self.interp_pt_y[1]
        self.interp_pt_logU_y = self.interp_pt_y[2]
        self.interp_pt_lfe_y = self.interp_pt_y[3]
        self.interp_pt_mu_y = self.interp_pt_y[4]

        self.interp_pt_logRho_z = self.interp_pt_z[0]
        self.interp_pt_logS_z = self.interp_pt_z[1]
        self.interp_pt_logU_z = self.interp_pt_z[2]

        self.interp_dt_logP_z = self.interp_dt_z[0]
        self.interp_dt_logS_z = self.interp_dt_z[1]

    def __call__(
        self, logT: ArrayLike, logP: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> NDArray:
        """__call__ method acting as convenience wrapper for the evaluate
        method. Calculates the equation of state output for the mixture.

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

    def __prepare(
        self, logT: ArrayLike, logP: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Prepare the equation of state input.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.
            X (ArrayLike): hydrogen mass-fraction.
            Z (ArrayLike): heavy-element mass-fraction.

        Returns:
            tuple[NDArray, NDArray, NDArray]: formated input and result arrays.
        """
        logT, logP = self.__check_pt(logT, logP)
        X, Y, Z = check_composition(X, Z)
        if logT.ndim > X.ndim:
            X = X * np.ones_like(logT)
            Y = Y * np.ones_like(logT)
            Z = Z * np.ones_like(logT)
        elif logT.ndim < X.ndim:
            logT = logT * np.ones_like(X)
            logP = logP * np.ones_like(X)

        self.X_close = np.isclose(X, 1, atol=self.eps1)
        self.Y_close = np.isclose(X, 1, atol=self.eps1)
        self.Z_close = np.isclose(Z, 1, atol=self.eps1)
        self.input_shape = logT.shape
        self.input_ndim = len(self.input_shape)
        res = get_zeros(input_shape=self.input_shape)
        return (logT, logP, X, Y, Z, res)

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

    def __check_pt(self, logT: ArrayLike, logP: ArrayLike) -> tuple[NDArray, NDArray]:
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
            tuple[NDArray, NDArray]: (logT, logP) as arrays.
        """

        if not isinstance(logT, np.ndarray):
            logT = np.array(logT, dtype=np.float64)
        if not isinstance(logP, np.ndarray):
            logP = np.array(logP, dtype=np.float64)

        if logT.shape != logP.shape:
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

        if np.all(self.X_close):
            logRho = self.interp_pt_logRho_x(logT, logP, grid=False)
        elif np.all(self.Y_close):
            logRho = self.interp_pt_logRho_y(logT, logP, grid=False)
        elif np.all(self.Z_close):
            logRho = self.interp_pt_logRho_z(logT, logP, grid=False)
        else:
            if np.any(self.X_close) and self.include_hhe_interactions:
                logRho_x = self.interp_pt_logRho_x(logT, logP, grid=False)
                logRho_x_eff = self.interp_pt_logRho_x_eff(logT, logP, grid=False)
                i = np.logical_and(self.tiny_val < X, self.tiny_val < Y)
                logRho_x[i] = logRho_x_eff[i]
                logRho_y = self.interp_pt_logRho_y(logT, logP, grid=False)
                logRho_z = self.interp_pt_logRho_z(logT, logP, grid=False)
            elif self.include_hhe_interactions:
                logRho_x = self.interp_pt_logRho_x_eff(logT, logP, grid=False)
            else:
                logRho_x = self.interp_pt_logRho_x(logT, logP, grid=False)
            logRho_y = self.interp_pt_logRho_y(logT, logP, grid=False)
            logRho_z = self.interp_pt_logRho_z(logT, logP, grid=False)

            iml = ideal_mixing_law(10**logRho_x, 10**logRho_y, 10**logRho_z, X, Y, Z)
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

        logRho = self.interp_pt_logRho_x(logT, logP, **self.kwargs)
        logS = self.interp_pt_logS_x(logT, logP, **self.kwargs)
        logU = self.interp_pt_logU_x(logT, logP, **self.kwargs)

        dlRho_dlT_P = self.interp_pt_logRho_x(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = self.interp_pt_logRho_x(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        dlS_dlT_P = self.interp_pt_logS_x(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = self.interp_pt_logS_x(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        lfe = self.interp_pt_lfe_x(logT, logP, **self.kwargs)
        mu = self.interp_pt_mu_x(logT, logP, **self.kwargs)

        input_shape = logT.shape
        res_x = get_zeros(input_shape)
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

    def __evaluate_x_eff(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for effective hydrogen.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """

        logRho = self.interp_pt_logRho_x_eff(logT, logP, **self.kwargs)
        logS = self.interp_pt_logS_x_eff(logT, logP, **self.kwargs)
        logU = self.interp_pt_logU_x_eff(logT, logP, **self.kwargs)

        dlRho_dlT_P = self.interp_pt_logRho_x_eff(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = self.interp_pt_logRho_x_eff(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        dlS_dlT_P = self.interp_pt_logS_x_eff(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = self.interp_pt_logS_x_eff(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        lfe = self.interp_pt_lfe_x_eff(logT, logP, **self.kwargs)
        mu = self.interp_pt_mu_x_eff(logT, logP, **self.kwargs)

        input_shape = logT.shape
        res_x_eff = get_zeros(input_shape)
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

    def __evaluate_y(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for helium.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """

        logRho = self.interp_pt_logRho_y(logT, logP, **self.kwargs)
        logS = self.interp_pt_logS_y(logT, logP, **self.kwargs)
        logU = self.interp_pt_logU_y(logT, logP, **self.kwargs)

        dlRho_dlT_P = self.interp_pt_logRho_y(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = self.interp_pt_logRho_y(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        dlS_dlT_P = self.interp_pt_logS_y(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = self.interp_pt_logS_y(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        lfe = self.interp_pt_lfe_y(logT, logP, **self.kwargs)
        mu = self.interp_pt_mu_y(logT, logP, **self.kwargs)

        input_shape = logT.shape
        res_y = get_zeros(input_shape)
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
        """Calculates equation of state output for the heavy element.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """
        logRho = self.interp_pt_logRho_z(logT, logP, **self.kwargs)
        logS = self.interp_pt_logS_z(logT, logP, **self.kwargs)
        logU = self.interp_pt_logU_z(logT, logP, **self.kwargs)

        dlRho_dlT_P = self.interp_pt_logRho_z(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = self.interp_pt_logRho_z(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        dlS_dlT_P = self.interp_pt_logS_z(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = self.interp_pt_logS_z(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        input_shape = logT.shape
        res_z = get_zeros(input_shape)
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
            NDArray: reduced equation of state output. The indices of the
                individual quantities are defined in the __init__ method.
        """
        logT, logP, X, Y, Z, res = self.__prepare(logT, logP, X, Z)

        if np.all(self.X_close) or not self.include_hhe_interactions:
            res_x = self.__evaluate_x(logT, logP)
        elif np.any(self.X_close) and self.include_hhe_interactions:
            i_x_eff = X < 1
            i_x = ~i_x_eff
            logT_x = logT[i_x]
            logP_x = logP[i_x]
            logT_x_eff = logT[i_x_eff]
            logP_x_eff = logP[i_x_eff]
            res_x = get_zeros(self.input_shape)
            res_x[:, i_x] = self.__evaluate_x(logT_x, logP_x)
            res_x[:, i_x_eff] = self.__evaluate_x_eff(logT_x_eff, logP_x_eff)
        else:
            res_x = self.__evaluate_x_eff(logT, logP)
        res_y = self.__evaluate_y(logT, logP)
        res_z = self.__evaluate_z(logT, logP)

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
        for log_entropy in [logS_x, logS_y, logS_z]:
            np.nan_to_num(
                log_entropy,
                copy=False,
                nan=np.nan,
                posinf=self.upper_logS,
                neginf=self.lower_logS,
        )

        S_x = 10**logS_x
        S_y = 10**logS_y
        S_z = 10**logS_z
        S = X * S_x + Y * S_y + Z * S_z
        S = S + get_mixing_entropy(Y=Y, Z=Z, A_z=self.A)
        logS = np.log10(S)
        logS = np.clip(logS, a_min=self.lower_logS, a_max=self.upper_logS)

        logU_x = res_x[self.i_logU]
        logU_y = res_y[self.i_logU]
        logU_z = res_z[self.i_logU]
        U = X * (10**logU_x) + Y * (10**logU_y) + Z * (10**logU_z)
        logU = np.log10(U)
        logU = np.nan_to_num(
            logU, nan=np.nan, posinf=self.upper_logU, neginf=self.lower_logU
        )
        logU = np.clip(logU, a_min=self.lower_logU, a_max=self.upper_logU)

        if np.all(self.X_close) or not self.include_hhe_interactions:
            dlS_dlT_P_x = self.interp_pt_logS_x(logT, logP, dx=1, **self.kwargs)
            dlS_dlP_T_x = self.interp_pt_logS_x(logT, logP, dy=1, **self.kwargs)
        elif np.any(self.X_close) and self.include_hhe_interactions:
            dlS_dlP_T_x = np.zeros_like(logS)
            dlS_dlT_P_x[i_x] = self.interp_pt_logS_x(
                logT_x, logP_x, dx=1, **self.kwargs
            )
            dlS_dlT_P_x[i_x_eff] = self.interp_pt_logS_x_eff(
                logT_x_eff, logP_x_eff, dx=1, **self.kwargs
            )
            dlS_dlP_T_x[i_x] = self.interp_pt_logS_x(
                logT_x, logP_x, dy=1, **self.kwargs
            )
            dlS_dlP_T_x[i_x_eff] = self.interp_pt_logS_x_eff(
                logT_x_eff, logP_x_eff, dy=1, **self.kwargs
            )
        else:
            dlS_dlT_P_x = self.interp_pt_logS_x_eff(logT, logP, dx=1, **self.kwargs)
            dlS_dlP_T_x = self.interp_pt_logS_x_eff(logT, logP, dy=1, **self.kwargs)

        dlS_dlT_P_y = self.interp_pt_logS_y(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T_y = self.interp_pt_logS_y(logT, logP, dy=1, **self.kwargs)
        dlS_dlT_P_z = self.interp_pt_logS_z(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T_z = self.interp_pt_logS_z(logT, logP, dy=1, **self.kwargs)

        dlS_dlP_T = (
            X * S_x * dlS_dlP_T_x + Y * S_y * dlS_dlP_T_y + Z * S_z * dlS_dlP_T_z
        ) / S
        dlS_dlT_P = (
            X * S_x * dlS_dlT_P_x + Y * S_y * dlS_dlT_P_y + Z * S_z * dlS_dlT_P_z
        ) / S

        if self.input_ndim > 0:
            shape = (3, 3) + logT.shape
            fac = np.zeros(shape)
            iX = np.isclose(X, self.tiny_val, atol=self.eps1)
            iY = np.isclose(Y, self.tiny_val, atol=self.eps1)
            iZ = np.isclose(Z, self.tiny_val, atol=self.eps1)
            if np.any(iX) or np.any(iY) or np.any(iZ):
                i = self.tiny_val < X
                fac[0, 0, i] = X[i] / rho_x[i] / res_x[self.i_chiRho, i]
                fac[0, 1, i] = -fac[0, 0, i] * res_x[self.i_chiT, i]
                fac[0, 2, i] = X[i] / res_x[self.i_mu, i]

                i = self.tiny_val < Y
                fac[1, 0, i] = Y[i] / rho_y[i] / res_y[self.i_chiRho, i]
                fac[1, 1, i] = -fac[1, 0, i] * res_y[self.i_chiT, i]
                fac[1, 2, i] = Y[i] / res_y[self.i_mu, i]

                i = self.tiny_val < Z
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
            if self.tiny_val < X:
                fac[0, 0] = X / rho_x / res_x[self.i_chiRho]
                fac[0, 1] = -fac[0, 0] * res_x[self.i_chiT]
                fac[0, 2] = X / res_x[self.i_mu]
            if self.tiny_val < Y:
                fac[1, 0] = Y / rho_y / res_y[self.i_chiRho]
                fac[1, 1] = -fac[1, 0] * res_y[self.i_chiT]
                fac[1, 2] = Y / res_y[self.i_mu]
            if self.tiny_val < Z:
                fac[2, 0] = Z / rho_z / res_z[self.i_chiRho]
                fac[2, 1] = -fac[2, 0] * res_z[self.i_chiT]
                fac[2, 2] = Z / res_z[self.i_mu]
        dlRho_dlP_T = rho * (fac[0, 0] + fac[1, 0] + fac[2, 0])
        dlRho_dlT_P = rho * (fac[0, 1] + fac[1, 1] + fac[2, 1])
        mu = 1 / (fac[0, 2] + fac[1, 2] + fac[2, 2])

        grad_ad = -dlS_dlP_T / dlS_dlT_P
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T
        check_grad_ad = np.isnan(grad_ad)
        if np.any(check_grad_ad):
            if self.input_ndim > 0:
                grad_ad[check_grad_ad] = self.lower_grad_ad
            else:
                grad_ad = self.lower_grad_ad
        grad_ad = np.clip(a=grad_ad, a_min=self.lower_grad_ad, a_max=self.upper_grad_ad)
        chiRho = np.clip(a=chiRho, a_min=self.lower_chiRho, a_max=self.upper_chiRho)
        chiT = np.clip(a=chiT, a_min=self.lower_chiT, a_max=self.upper_chiT)
        gamma3 = 1 - (grad_ad / chiRho)
        gamma1 = chiT * (gamma3 - 1) + chiRho

        # alternative expressions:
        # gamma1 = chiRho / (1 - chiT * grad_ad)
        # gamma3 = 1 + gamma1 * grad_ad

        # from the definition of the specific heat
        # cp = S * dlS_dlT_P
        # Alternatively from Stellar Interiors pp. 176
        cp = P * chiT / (rho * T * chiRho * grad_ad)
        if self.input_ndim > 0:
            i = gamma1 >= self.tiny_val
            cv = np.zeros_like(logT)
            cv[i] = cp[i] * chiRho[i] / gamma1[i]
            cv[~i] = cp[~i]
            # Alternatively from Chabrier et al. (2019) eq. 5:
            # cv = cp - (P * chiT**2) / (rho * T * chiRho)
            c_sound = self.tiny_val * np.ones_like(logT)
            c_sound[i] = np.sqrt(P[i] / rho[i] * gamma1[i])
        else:
            if gamma1 >= self.tiny_val:
                cv = cp * chiRho / gamma1
                c_sound = np.sqrt(P / rho * gamma1)
            else:
                cv = cp
                c_sound = self.tiny_val

        # these are at constant density or temperature
        dS_dT = cv / T  # definition of specific heat
        dS_dRho = -(P / T / rho**2) * chiT  # maxwell relation
        dE_dRho = (P / rho**2) * (1 - chiT)

        # only hydrogen and helium contribute to free electrons
        lfe = np.log10(X * 10 ** res_x[self.i_lfe] + Y * 10 ** res_y[self.i_lfe])
        lfe = np.clip(a=lfe, a_min=-99, a_max=None)
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
