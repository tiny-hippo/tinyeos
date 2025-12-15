import os
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root_scalar
from scipy.optimize.elementwise import bracket_root, find_root

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
from tinyeos.tinypteos import TinyPT


class TinyDT:
    """
    Equation of state for mixtures of hydrogen, helium, and heavy elements.
    This class provides a high-level interface for calculating the equation of state
    properties for mixtures of hydrogen, helium, and heavy elements. It wraps
    the TinyPT equation of state and handles root-finding to match density and pressure.

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
    __call__(logT, logRho, X, Z)
        Convenience wrapper for "evaluate". Calculates equation of state output
        for the mixture.
    evaluate(logT, logRho, X, Z)
        Calculates equation of state output for the mixture.
        Returns an NDArray with equation of state quantities.
    __helper(logP, logT, logRho, X, Y, Z)
        Internal helper for root finding. Returns difference
        between input and calculated density.
    __root_finder(logT, logRho, X, Y, Z)
        Finds the pressure corresponding to the input density using root-finding.
    
    Notes
    -----
    - The output NDArray indices are defined in definitions.py.
    - If calculation fails, results are set to NaN.
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
        """__init__ method.

        Defines parameters and either loads or builds the interpolants.

        Args:
        ----
            which_hhe (str, optional): hydrogen-helium equation of state
                to use. Options are "cms", "scvh" or "scvh_extended". Defaults to "cms".
            which_heavy : str, optional
                Heavy-element tables to use. Options: "h2o", "sesame_h2o", "aqua", 
                "sio2", "fe", "co", and "mixture". Default: "h2o".
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
        ------
            NotImplementedError: raised if which_heavy or which_hhe choices.

        """
        self.tpt = TinyPT(
            which_hhe=which_hhe,
            which_heavy=which_heavy,
            Z1=Z1,
            Z2=Z2,
            Z3=Z3,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
            build_interpolants=build_interpolants,
        )

        set_eos_params(
            eos=self, which_hhe=which_hhe, which_heavy=which_heavy, Z1=Z1, Z2=Z2, Z3=Z3
        )

    def __call__(
        self, logT: ArrayLike, logRho: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> NDArray:
        """__call__ method acting as convenience wrapper for the evaluate method.

        Calculates the equation of state output for the mixture.
        If the calculation fails the results are set to nans.

        Args:
        ----
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            X (ArrayLike): hydrogen mass fraction.
            Z (ArrayLike): heavy-element mass fraction.

        Returns:
        -------
            NDArray: equation of state output. The indices of the
                individual quantities are defined in definitions.py.

        """
        return self.evaluate(logT=logT, logRho=logRho, X=X, Z=Z)

    def __helper(
        self,
        logP: ArrayLike,
        logT: ArrayLike,
        logRho: ArrayLike,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
    ) -> ArrayLike:
        """Helper function for the root finding.

        Calls the pressure-temperature equation of state and returns
        the difference between the input and the calculated density.

        Args:
        ----
            logP (ArrayLike): log10 of the pressure.
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            X (ArrayLike): hydrogen mass fraction.
            Y (ArrayLike): helium mass fraction.
            Z (ArrayLike): heavy-element mass fraction.

        Returns:
        -------
            ArrayLike: difference of the input and calculated density.

        """
        logRho_iml = self.tpt._TinyPT__ideal_mixture(
            logT=logT, logP=logP, X=X, Y=Y, Z=Z
        )
        return logRho - logRho_iml

    def __root_finder(
        self,
        logT: ArrayLike,
        logRho: ArrayLike,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
    ) -> tuple[bool, ArrayLike]:
        """Root-finding function.

        Uses the pressure-temperature equation of state
        to find the pressure corresponding to the input density.

        Args:
        ----
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            X (ArrayLike): hydrogen mass fraction.
            Y (ArrayLike): helium mass fraction.
            Z (ArrayLike): heavy-element mass fraction.

        Returns:
        -------
            tuple[bool, ArrayLike]: root-finding result.

        """
        if logT.ndim == 0:
            f1 = self.__helper(
                logP=self.logP_min, logT=logT, logRho=logRho, X=X, Y=Y, Z=Z
            )
            f2 = self.__helper(
                logP=self.logP_max, logT=logT, logRho=logRho, X=X, Y=Y, Z=Z
            )
            if np.sign(f1) == np.sign(f2):
                success = False
                logP = np.nan
            else:
                sol = root_scalar(
                    f=self.__helper,
                    args=(logT, logRho, X, Y, Z),
                    method="brentq",
                    bracket=[self.tpt.logP_min, self.tpt.logP_max],
                )
                success = sol.converged
                logP = sol.root
        else:
            logP = np.empty(shape=logT.shape)
            logP.fill(np.nan)
            res_bracket = bracket_root(
                f=self.__helper,
                xl0=self.logP_min,
                xr0=self.logP_max,
                xmin=self.logP_min,
                xmax=self.logP_max,
                args=(logT, logRho, X, Y, Z),
            )
            res_root = find_root(
                self.__helper,
                res_bracket.bracket,
                args=(logT, logRho, X, Y, Z),
            )
            success = res_root.success
            logP[success] = res_root.x[success]
        return (success, logP)

    def evaluate(
        self, logT: ArrayLike, logRho: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> NDArray:
        """Calculate the equation of state output for the mixture.

        Calculates the equation of state output for the mixture.
        If the calculation fails the results are set to nans.

        Args:
        ----
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            X (ArrayLike): hydrogen mass fraction.
            Z (ArrayLike): heavy-element mass fraction.

        Returns:
        -------
            NDArray: equation of state output. The indices of the
                individual quantities are defined in definitions.py.

        """
        dummy = 6 * np.ones_like(logT)
        logT, _, X, Y, Z, res = self.tpt._TinyPT__prepare(
            logT=logT, logP=dummy, X=X, Z=Z
        )
        if not isinstance(logRho, np.ndarray):
            logRho = np.array(logRho, dtype=np.float64)
        success, logP = self.__root_finder(logT=logT, logRho=logRho, X=X, Y=Y, Z=Z)
        if logT.ndim == 0:
            if success:
                res[...] = self.tpt.evaluate(logT=logT, logP=logP, X=X, Z=Z)
            else:
                res[self.i_logT] = logT
                res[self.i_logRho] = logRho
                res[self.i_logRho + 1 :] = np.nan
        else:
            if np.all(success):
                res[...] = self.tpt.evaluate(logT=logT, logP=logP, X=X, Z=Z)
            else:
                res[:, success] = self.tpt.evaluate(
                    logT=logT[success], logP=logP[success], X=X[success], Z=Z[success]
                )
                not_success = np.invert(success)

                res[self.i_logT, not_success] = logT[not_success]
                res[self.i_logRho, not_success] = logRho[not_success]
                res[self.i_logRho + 1 :, not_success] = np.nan
        return res


class TinyDirectDT(InterpolantsBuilder):
    """
    Equation of state for mixtures of hydrogen, helium, and heavy elements.
    Mixtures are calculated using the density-temperature tables.

    Parameters
    ----------
    which_hhe : str, optional
        Hydrogen-helium equation of state to use. Options: "cms", "scvh",
        and "scvh_extended". Default is "cms".
    which_heavy : str, optional
        Heavy-element equation of state to use. Options: "h2o", "aqua", "sio2",
        "fe", "co", and "mixture". Default is "h2o".
    Z1 : float, optional
        Mass fraction of H2O in the heavy-element mixture. Default: 0.5.
    Z2 : float, optional
        Mass fraction of SiO2 in the heavy-element mixture. Default: 0.5.
    Z3 : float, optional
        Mass fraction of Fe in the heavy-element mixture. Default: 0.0.
    include_hhe_interactions : bool, optional
        Whether to include hydrogen-helium interactions. Default is False.
    use_smoothed_xy_tables : bool, optional
        Use smoothed hydrogen and helium tables. Default is False.
    use_smoothed_z_tables : bool, optional
        Use smoothed heavy-element tables. Default is False.
    build_interpolants : bool, optional
        Whether to build interpolants. Default is False.

    Raises
    ------
    NotImplementedError
        Raised if which_heavy or which_hhe are invalid choices.

    Methods
    -------
    __call__(logT, logRho, X, Z)
        Convenience wrapper for `evaluate`. Returns equation of state output
        for the mixture.
    __prepare(logT, logRho, X, Z)
        Prepares and formats input arrays for equation of state calculations.
    __load_interp(filename)
        Loads interpolant data from disk.
    __check_dt(logT, logRho)
        Checks that temperature and density inputs are within equation of state limits.
    __ideal_mixture(logT, logRho, X, Y, Z)
        Calculates individual densities and pressure for the mixture
        using the ideal mixing law.
    __ideal_mixing_law_wrapper(logP, logT, logRho, X, Y, Z)
        Helper for root finding in the ideal mixing law.
    __evaluate_x(logT, logRho)
        Calculates equation of state output for hydrogen.
    __evaluate_x_eff(logT, logRho)
        Calculates equation of state output for effective hydrogen (with interactions).
    __evaluate_y(logT, logRho)
        Calculates equation of state output for helium.
    __evaluate_z(logT, logRho)
        Calculates equation of state output for the heavy element.
    evaluate(logT, logRho, X, Z, verbose=False)
        Calculates the equation of state output for the mixture,
        supporting only scalar inputs at this time.

    Notes
    -----
    - The indices of the output quantities are defined in the class.
    - If root finding fails, output is filled with NaNs.
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
        # to-do: scvh is currently giving inconsistent results between
        # TinyPT and TinyDT; needs to be fixed before allowing it again
        if which_hhe == "scvh":
            raise NotImplementedError("scvh is currently disabled")
        self.include_hhe_interactions = include_hhe_interactions
        if include_hhe_interactions and which_hhe == "scvh":
            raise NotImplementedError("can't include H-He interactions with scvh")

        if build_interpolants:
            super().__init__()

        set_eos_params(
            eos=self, which_hhe=which_hhe, which_heavy=which_heavy, Z1=Z1, Z2=Z2, Z3=Z3
        )

        self.tpt = TinyPT(
            which_heavy=which_heavy,
            which_hhe=which_hhe,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
            build_interpolants=build_interpolants,
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

        self.interp_dt_x = self.__load_interp("interp_dt_x_" + which_hhe + ".npy")
        self.interp_dt_y = self.__load_interp("interp_dt_y_" + which_hhe + ".npy")
        self.interp_dt_z = self.__load_interp(
            "interp_dt_z_" + self.which_heavy + ".npy"
        )

        if self.include_hhe_interactions:
            self.interp_pt_x_eff = self.__load_interp(
                "interp_pt_x_eff_" + which_hhe + ".npy"
            )
            self.interp_dt_x_eff = self.__load_interp(
                "interp_dt_x_eff_" + which_hhe + ".npy"
            )
            self.interp_pt_xy = self.__load_interp("interp_pt_xy_int.npy")

        self.interp_dt_logP_x = self.interp_dt_x[0]
        self.interp_dt_logS_x = self.interp_dt_x[1]
        self.interp_dt_logU_x = self.interp_dt_x[2]
        self.interp_dt_lfe_x = self.interp_dt_x[3]
        self.interp_dt_mu_x = self.interp_dt_x[4]

        if self.include_hhe_interactions:
            self.interp_dt_logP_x_eff = self.interp_dt_x_eff[0]
            self.interp_dt_logS_x_eff = self.interp_dt_x_eff[1]
            self.interp_dt_logU_x_eff = self.interp_dt_x_eff[2]
            self.interp_dt_lfe_x_eff = self.interp_dt_x_eff[3]
            self.interp_dt_mu_x_eff = self.interp_dt_x_eff[4]
            self.interp_pt_V_mix_xy = self.interp_pt_xy[0]
            self.interp_pt_S_mix_xy = self.interp_pt_xy[1]

        self.interp_dt_logP_y = self.interp_dt_y[0]
        self.interp_dt_logS_y = self.interp_dt_y[1]
        self.interp_dt_logU_y = self.interp_dt_y[2]
        self.interp_dt_lfe_y = self.interp_dt_y[3]
        self.interp_dt_mu_y = self.interp_pt_y[4]

        self.interp_dt_logP_z = self.interp_dt_z[0]
        self.interp_dt_logS_z = self.interp_dt_z[1]
        self.interp_dt_logU_z = self.interp_dt_z[2]

        self.interp_pt_logRho_x = self.interp_pt_x[0]
        self.interp_pt_logS_x = self.interp_pt_x[1]
        if self.include_hhe_interactions:
            self.interp_pt_logRho_x_eff = self.interp_pt_x_eff[0]
            self.interp_pt_logS_x_eff = self.interp_pt_x_eff[1]
        self.interp_pt_logRho_y = self.interp_pt_y[0]
        self.interp_pt_logS_y = self.interp_pt_y[1]
        self.interp_pt_logRho_z = self.interp_pt_z[0]
        self.interp_pt_logS_z = self.interp_pt_z[1]
        self.interp_pt_logU_z = self.interp_pt_z[2]

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

    def __prepare(
        self, logT: ArrayLike, logRho: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Prepare the equation of state input.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (float): log10 of the density.
            X (ArrayLike): hydrogen mass-fraction.
            Z (ArrayLike): heavy-element mass-fraction.

        Returns:
            tuple[NDArray, NDArray, NDArray]: formated input and result arrays.
        """
        logT, logRho = self.__check_dt(logT, logRho)
        X, Y, Z = check_composition(X, Z)
        if logT.ndim > X.ndim:
            X = X * np.ones_like(logT)
            Y = Y * np.ones_like(logT)
            Z = Z * np.ones_like(logT)
        elif logT.ndim < X.ndim:
            logT = logT * np.ones_like(X)
            logRho = logRho * np.ones_like(X)

        self.X_close = np.isclose(X, 1, atol=self.eps1)
        self.Y_close = np.isclose(X, 1, atol=self.eps1)
        self.Z_close = np.isclose(Z, 1, atol=self.eps1)
        self.input_shape = logT.shape
        self.input_ndim = len(self.input_shape)
        res = get_zeros(input_shape=self.input_shape)
        return (logT, logRho, X, Y, Z, res)

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

    def __check_dt(self, logT: ArrayLike, logRho: ArrayLike) -> None:
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
            tuple[NDArray, NDArray]: (logT, logRho) as arrays.
        """

        if not isinstance(logT, np.ndarray):
            logT = np.array(logT, dtype=np.float64)
        if not isinstance(logRho, np.ndarray):
            logRho = np.array(logRho, dtype=np.float64)

        if logT.shape != logRho.shape:
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

    def __ideal_mixture(
        self,
        logT: float,
        logRho: float,
        X: float,
        Y: float,
        Z: float,
    ) -> tuple:
        """Calculates the individual densities of the elements in the mixture.

        Args:
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            X (float): hydrogen mass-fraction
            Y (float): helium mass-fraction.
            Z (float): heavy-element mass-fraction.

        Returns:
            tuple: tuple consisting of convergence information, individual
                densities and gas pressure.
        """
        if np.isclose(Z, 1, atol=self.eps1):
            conv = True
            logP = self.interp_dt_logP_z(logT, logRho, **self.kwargs)
            logRho_x = self.tiny_logRho
            logRho_y = self.tiny_logRho
            logRho_z = logRho
        elif np.isclose(X, 1, atol=self.eps1):
            conv = True
            logP = self.interp_dt_logP_x(logT, logRho, **self.kwargs)
            logRho_x = logRho
            logRho_y = self.tiny_logRho
            logRho_z = self.tiny_logRho
        elif np.isclose(Y, 1, atol=self.eps1):
            conv = True
            logP = self.interp_dt_logP_y(logT, logRho, **self.kwargs)
            logRho_x = self.tiny_logRho
            logRho_y = logRho
            logRho_z = self.tiny_logRho
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
                if (
                    self.tiny_val < X
                    and self.tiny_val < Y
                    and self.include_hhe_interactions
                ):
                    logRho_x = self.interp_pt_logRho_x_eff(logT, logP, **self.kwargs)
                else:
                    logRho_x = self.interp_pt_logRho_x(logT, logP, **self.kwargs)
                logRho_y = self.interp_pt_logRho_y(logT, logP, **self.kwargs)
                logRho_z = self.interp_pt_logRho_z(logT, logP, **self.kwargs)

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
        if self.tiny_val < X and self.tiny_val < Y and self.include_hhe_interactions:
            logRho_x = self.interp_pt_logRho_x_eff(logT, logP, **self.kwargs)
        elif self.tiny_val < X:
            logRho_x = self.interp_pt_logRho_x(logT, logP, **self.kwargs)
        else:
            logRho_x = self.tiny_logRho
        if self.tiny_val < Y:
            logRho_y = self.interp_pt_logRho_y(logT, logP, **self.kwargs)
        else:
            logRho_y = self.tiny_logRho
        if self.tiny_val < Z:
            logRho_z = self.interp_pt_logRho_z(logT, logP, **self.kwargs)
        else:
            logRho_z = self.tiny_logRho

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

        logP = self.interp_dt_logP_x(logT, logRho, **self.kwargs)
        # use (logT, logP) for logS to be consistent with the derivatives
        logS = self.interp_pt_logS_x(logT, logP, **self.kwargs)
        logU = self.interp_dt_logU_x(logT, logRho, **self.kwargs)

        chiT = self.interp_dt_logP_x(logT, logRho, dx=1, **self.kwargs)
        chiRho = self.interp_dt_logP_x(logT, logRho, dy=1, **self.kwargs)

        dlS_dlT_P = self.interp_pt_logS_x(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = self.interp_pt_logS_x(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        lfe = self.interp_dt_lfe_x(logT, logRho, **self.kwargs)
        mu = self.interp_dt_mu_x(logT, logRho, **self.kwargs)

        input_shape = logT.shape
        res_x = get_zeros(input_shape=input_shape)
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

        logP = self.interp_dt_logP_x_eff(logT, logRho, **self.kwargs)
        # use (logT, logP) for logS to be consistent with the derivatives
        logS = self.interp_pt_logS_x_eff(logT, logP, **self.kwargs)
        logU = self.interp_dt_logU_x_eff(logT, logRho, **self.kwargs)

        chiT = self.interp_dt_logP_x_eff(logT, logRho, dx=1, **self.kwargs)
        chiRho = self.interp_dt_logP_x_eff(logT, logRho, dy=1, **self.kwargs)

        dlS_dlT_P = self.interp_pt_logS_x_eff(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = self.interp_pt_logS_x_eff(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        lfe = self.interp_dt_lfe_x_eff(logT, logRho, **self.kwargs)
        mu = self.interp_dt_mu_x_eff(logT, logRho, **self.kwargs)

        input_shape = logT.shape
        res_x_eff = get_zeros(input_shape=input_shape)
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

        logP = self.interp_dt_logP_y(logT, logRho, **self.kwargs)
        # use (logT, logP) for logS to be consistent with the derivatives
        logS = self.interp_pt_logS_y(logT, logP, **self.kwargs)
        logU = self.interp_dt_logU_y(logT, logRho, **self.kwargs)

        chiT = self.interp_dt_logP_y(logT, logRho, dx=1, **self.kwargs)
        chiRho = self.interp_dt_logP_y(logT, logRho, dy=1, **self.kwargs)

        dlS_dlT_P = self.interp_pt_logS_y(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = self.interp_pt_logS_y(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        lfe = self.interp_dt_lfe_y(logT, logRho, **self.kwargs)
        mu = self.interp_dt_mu_y(logT, logRho, **self.kwargs)

        input_shape = logT.shape
        res_y = get_zeros(input_shape=input_shape)
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
        use_legacy_methods = False
        logP = self.interp_dt_logP_z(logT, logRho, **self.kwargs)
        # use (logT, logP) for logS to be consistent with the derivatives
        logS = self.interp_pt_logS_z(logT, logP, **self.kwargs)
        logU = self.interp_dt_logU_z(logT, logRho, **self.kwargs)

        # new method with (logT, logP)
        dlRho_dlP_T = self.interp_pt_logRho_z(logT, logP, dy=1, **self.kwargs)
        dlRho_dlT_P = self.interp_pt_logRho_z(logT, logP, dx=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        chiT = self.interp_dt_logP_z(logT, logRho, dx=1, **self.kwargs)
        chiRho = self.interp_dt_logP_z(logT, logRho, dy=1, **self.kwargs)

        dlS_dlP_T = self.interp_pt_logS_z(logT, logP, dy=1, **self.kwargs)
        dlS_dlT_P = self.interp_pt_logS_z(logT, logP, dx=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        # old method with (logT, logRho)
        if use_legacy_methods:
            dlRho_dlP_T = self.interp_pt_logRho_z(logT, logP, dy=1, **self.kwargs)
            dlRho_dlT_P = self.interp_pt_logRho_z(logT, logP, dx=1, **self.kwargs)
            chiRho = 1 / dlRho_dlP_T
            chiT = -dlRho_dlT_P / dlRho_dlP_T

            dlS_dlT_rho = self.interp_dt_logS_z(logT, logRho, dx=1, **self.kwargs)
            dlS_dlRho_T = self.interp_dt_logS_z(logT, logRho, dy=1, **self.kwargs)
            dlS_dlRho_P = dlS_dlRho_T + dlS_dlT_rho * (-chiRho / chiT)
            dlS_dlP_rho = dlS_dlT_rho / chiT
            gamma1 = -dlS_dlRho_P / dlS_dlP_rho
            grad_ad = (1 - chiRho / gamma1) / chiT
            # alternatively:
            # grad_ad = 1 / (chiT - dlS_dlT_rho * chiRho / dlS_dlRho_T)

        input_shape = logT.shape
        res_z = get_zeros(input_shape=input_shape)
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

        logT, logRho, X, Y, Z, res = self.__prepare(logT=logT, logRho=logRho, X=X, Z=Z)

        iml = self.__ideal_mixture(logT, logRho, X, Y, Z)
        if not iml[0]:
            if verbose:
                msg = "Evaluate failed in density root find at"
                print(msg, f"logT = {logT}, logRho = {logRho}")
            res[self.i_logT] = logT
            res[self.i_logRho] = logRho
            res[self.i_logP :] = np.nan
            return res

        logRho_x = np.array(iml[1], dtype=np.float64)
        logRho_y = np.array(iml[2], dtype=np.float64)
        logRho_z = np.array(iml[3], dtype=np.float64)
        logP = iml[4]

        if (
            np.all(self.X_close)
            or np.all(self.Y_close)
            or np.all(self.Z_close)
            or not self.include_hhe_interactions
        ):
            res_x = self.__evaluate_x(logT, logRho_x)
        elif np.any(self.X_close) and self.include_hhe_interactions:
            i_x_eff = X < 1
            i_x = ~i_x_eff
            logT_x = logT[i_x]
            logP_x = logP[i_x]
            logRho_x_normal = logRho_x[i_x]
            logT_x_eff = logT[i_x_eff]
            logP_x_eff = logP[i_x_eff]
            logRho_x_eff = logRho_x[i_x_eff]
            res_x = get_zeros(input_shape=self.input_shape)
            res_x[:, i_x] = self.__evaluate_x(logT_x, logRho_x_normal)
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
        S = S + get_mixing_entropy(Y=Y, Z=Z, A_z=self.A)
        logS = np.log10(S)

        logU_x = res_x[self.i_logU]
        logU_y = res_y[self.i_logU]
        logU_z = res_z[self.i_logU]
        U = X * (10**logU_x) + Y * (10**logU_y) + Z * (10**logU_z)
        logU = np.log10(U)

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
