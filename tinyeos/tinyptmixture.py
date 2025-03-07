import numpy as np
from numpy.typing import ArrayLike, NDArray

from tinyeos.definitions import (
    atomic_masses,
    eps1,
    logP_max,
    logP_min,
    logT_max,
    logT_min,
    tiny_val,
)
from tinyeos.support import get_mixing_entropy, get_zeros
from tinyeos.tinypteos import TinyPT


class TinyPTMixture:
    """Temperature-pressure equation of state for a mixture of hydrogen,
    helium and three heavy elements. Units are cgs everywhere.

    Equations of state implemented:
        Hydrogen-Helium:
            CMS (Chabrier et al. 2019),
            SCvH (Saumon et al. 1995).
        Heavy element:
            H2O (QEOS, More et al. 1988 and AQUA, Haldemann et al. 2020),
            SiO2 (QEOS, More et al. 1988),
            Fe (QEOS, More et al. 1998),
            CO (QEOS, Podolak et al. 2022),
    """

    def __init__(
        self,
        which_xy: str = "cms",
        which_z1: str = "h2o",
        which_z2: str = "sio2",
        which_z3: str = "fe",
        include_hhe_interactions: bool = True,
        use_smoothed_xy_tables: bool = False,
        use_smoothed_z_tables: bool = False,
        limit_bad_values: bool = False,
    ):
        """__init__ method. Defines parameters and either loads or
        builds the interpolants.

        Args:
            which_xy (str, optional): which hydrogen-helium equation of state
                to use. Defaults to "cms". Options are "cms" or "scvh".
                Defaults to "cms".
            which_z1 (str, optional): which heavy-element equation of state
                to use. Options are "h2o", "aqua", "sio2", "mixture", "fe"
                or "co". Defaults to "h2o".
            which_z2 (str, optional): which heavy-element equation of state
                to use. Options are "h2o", "aqua", "sio2", "mixture", "fe"
                or "co". Defaults to "sio2".
            which_z3 (str, optional): which heavy-element equation of state
                to use. Options are "h2o", "aqua", "sio2", "mixture", "fe"
                or "co". Defaults to "fe".
            include_hhe_interactions (bool, optional): whether to include
                hydrogen-helium interactions. Defaults to True.
            use_smoothed_xy_tables (bool, optional): whether to use smoothed
                hydrogen and helium tables. Defaults to False.
            use_smoothed_z_tables (bool, optional): whether to use smoothed
                heavy-element tables. Defaults to False.
            limit_bad_values (bool, optional): whether to limit bad equation
                of state results. Defaults to False.
        """
        self.logP_max = logP_max
        self.logP_min = logP_min
        self.logT_max = logT_max
        self.logT_min = logT_min
        if which_xy == "scvh_extended":
            self.logP_min = -6.00
            self.logT_min = 1.10

        self.num_vals_for_evaluate = 7
        self.num_vals_for_return = 6
        self.i_logRho = 0
        self.i_logS = 1
        self.i_dlS_dlP = 2
        self.i_dlS_dlT = 3
        self.i_grad_ad = 4
        self.i_chiRho = 5
        self.i_chiT = 6
        self.include_hhe_interactions = include_hhe_interactions
        self.limit_bad_values = limit_bad_values
        self.kwargs = {"grid": False}

        # limits for derivatives
        self.lower_grad_ad = 0.01
        self.lower_chiT = 0.01
        self.lower_chiRho = 0.01
        self.upper_grad_ad = 2.5
        self.upper_chiT = 2.5
        self.upper_chiRho = 2.5

        # atomic weights
        self.A1 = atomic_masses[which_z1]
        self.A2 = atomic_masses[which_z2]
        self.A3 = atomic_masses[which_z3]

        self.tpt_z1 = TinyPT(
            which_hhe=which_xy,
            which_heavy=which_z1,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
        )

        self.tpt_z2 = TinyPT(
            which_hhe=which_xy,
            which_heavy=which_z2,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
        )

        self.tpt_z3 = TinyPT(
            which_hhe=which_xy,
            which_heavy=which_z3,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
        )
        self.tpt_zs = [self.tpt_z1, self.tpt_z2, self.tpt_z3]

    def __call__(
        self,
        logT: ArrayLike,
        logP: ArrayLike,
        X: ArrayLike,
        Z1: ArrayLike,
        Z2: ArrayLike,
        Z3: ArrayLike,
    ) -> NDArray:
        """__call__ method acting as convenience wrapper for the evaluate
        method. Calculates the equation of state output for the mixture.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.
            X (ArrayLike): hydrogen mass-fraction.
            Z1 (ArrayLike): first heavy-element mass-fraction.
            Z2 (ArrayLike): second heavy-element mass-fraction.
            Z3 (ArrayLike): third heavy-element mass-fraction.

        Returns:
            NDArray: logRho, logS, grad_ad, chiRho, chiT and c_sound.
        """
        return self.evaluate(logT, logP, X, Z1, Z2, Z3)

    def __check_PT(self, logT: ArrayLike, logP: ArrayLike) -> tuple[NDArray, NDArray]:
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

    def __check_composition(
        self, X: ArrayLike, Z1: ArrayLike, Z2: ArrayLike, Z3: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Checks whether input composition adds up to less than one,
        dumps the residual into helium, and formats the mass fractions.

        Args:
            X (ArrayLike): hydrogen mass fraction.
            Z1 (ArrayLike): first heavy-element mass fraction.
            Z2 (ArrayLike): second heavy-element mass fraction.
            Z3 (ArrayLike): third heavy-element mass fraction.

        Raises:
            ValueError: X, Z1, Z2 and Z3 must have equal shape
                and their sum must be smaller or equal 1.

        Returns:
            tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
                tuple of hydrogen, helium and heavy-element mass fractions.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Z1, np.ndarray):
            Z1 = np.array(Z1)
        if not isinstance(Z2, np.ndarray):
            Z2 = np.array(Z2)
        if not isinstance(Z3, np.ndarray):
            Z3 = np.array(Z3)

        ndims = set([X.ndim, Z1.ndim, Z2.ndim, Z3.ndim])
        if len(ndims) != 1:
            msg = "composition arrays must have the same dimensions."
            raise ValueError(msg)

        if not np.all(X + Z1 + Z2 + Z3 <= 1):
            msg = "sum of mass fractions must be less than one"
            raise ValueError(msg)

        Y = 1 - X - Z1 - Z2 - Z3
        composition = np.asarray([X, Y, Z1, Z2, Z3])
        check_zero = np.isclose(composition, 0, atol=eps1)
        composition[check_zero] = 0
        check_one = np.isclose(composition, 1, atol=eps1)
        composition[check_one] = 1
        X = composition[0]
        Y = composition[1]
        Z1 = composition[2]
        Z2 = composition[3]
        Z3 = composition[4]
        return (X, Y, Z1, Z2, Z3)

    def __unpack(
        self, res: NDArray
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Returns the results from the equation of state call
        as a tuple.

        Args:
            res (NDArray): results from the equation of state call.

        Returns:
            tuple[float, float, float, float, float]:
                (logRho, logS, dlS_dlP_T, dlS_dlT_P, grad_ad)
        """
        logRho = res[self.i_logRho]
        logS = res[self.i_logS]
        dlS_dlP_T = res[self.i_dlS_dlP]
        dlS_dlT_P = res[self.i_dlS_dlT]
        grad_ad = res[self.i_grad_ad]
        return (logRho, logS, dlS_dlP_T, dlS_dlT_P, grad_ad)

    def __evaluate_x(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for hydrogen.
        Only calculates the density, entropy, entropy derivatives,
        and adiabatic gradient.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """
        tpt = self.tpt_z1
        logRho = tpt.interp_pt_logRho_x(logT, logP, **self.kwargs)
        logS = tpt.interp_pt_logS_x(logT, logP, **self.kwargs)

        dlS_dlT_P = tpt.interp_pt_logS_x(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = tpt.interp_pt_logS_x(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        dlRho_dlT_P = tpt.interp_pt_logRho_x(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = tpt.interp_pt_logRho_x(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        input_shape = logT.shape
        res_x = get_zeros(input_shape=input_shape, num_vals=self.num_vals_for_evaluate)
        res_x[self.i_logRho] = logRho
        res_x[self.i_logS] = logS
        res_x[self.i_dlS_dlP] = dlS_dlP_T
        res_x[self.i_dlS_dlT] = dlS_dlT_P
        res_x[self.i_grad_ad] = grad_ad
        res_x[self.i_chiRho] = chiRho
        res_x[self.i_chiT] = chiT
        return res_x

    def __evaluate_x_eff(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for effective hydrogen.
        Only calculates the density, entropy, entropy derivatives,
        and adiabatic gradient.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """
        tpt = self.tpt_z1
        logRho = tpt.interp_pt_logRho_x_eff(logT, logP, **self.kwargs)
        logS = tpt.interp_pt_logS_x_eff(logT, logP, **self.kwargs)

        dlS_dlT_P = tpt.interp_pt_logS_x_eff(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = tpt.interp_pt_logS_x_eff(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        dlRho_dlT_P = tpt.interp_pt_logRho_x_eff(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = tpt.interp_pt_logRho_x_eff(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        input_shape = logT.shape
        res_x_eff = get_zeros(
            input_shape=input_shape, num_vals=self.num_vals_for_evaluate
        )
        res_x_eff[self.i_logRho] = logRho
        res_x_eff[self.i_logS] = logS
        res_x_eff[self.i_dlS_dlP] = dlS_dlP_T
        res_x_eff[self.i_dlS_dlT] = dlS_dlT_P
        res_x_eff[self.i_grad_ad] = grad_ad
        res_x_eff[self.i_chiRho] = chiRho
        res_x_eff[self.i_chiT] = chiT
        return res_x_eff

    def __evaluate_y(self, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for helium.
        Only calculates the density, entropy, entropy derivatives,
        and adiabatic gradient.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """
        tpt = self.tpt_z1
        logRho = tpt.interp_pt_logRho_y(logT, logP, **self.kwargs)
        logS = tpt.interp_pt_logS_y(logT, logP, **self.kwargs)

        dlS_dlT_P = tpt.interp_pt_logS_y(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = tpt.interp_pt_logS_y(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        dlRho_dlT_P = tpt.interp_pt_logRho_y(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = tpt.interp_pt_logRho_y(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        input_shape = logT.shape
        res_y = get_zeros(input_shape=input_shape, num_vals=self.num_vals_for_evaluate)
        res_y[self.i_logRho] = logRho
        res_y[self.i_logS] = logS
        res_y[self.i_dlS_dlP] = dlS_dlP_T
        res_y[self.i_dlS_dlT] = dlS_dlT_P
        res_y[self.i_grad_ad] = grad_ad
        res_y[self.i_chiRho] = chiRho
        res_y[self.i_chiT] = chiT
        return res_y

    def __evaluate_z(self, which_iz: int, logT: ArrayLike, logP: ArrayLike) -> NDArray:
        """Calculates equation of state output for the heavy element. Only
        calculates the density, entropy, entropy derivatives,
        and adiabatic gradient.

        Args:
            which_iz (int): which instance of TinyPT to use.
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.

        Returns:
            NDArray: equation of state output.
        """
        tpt_zi = self.tpt_zs[which_iz]
        logRho = tpt_zi.interp_pt_logRho_z(logT, logP, **self.kwargs)
        logS = tpt_zi.interp_pt_logS_z(logT, logP, **self.kwargs)

        dlS_dlT_P = tpt_zi.interp_pt_logS_z(logT, logP, dx=1, **self.kwargs)
        dlS_dlP_T = tpt_zi.interp_pt_logS_z(logT, logP, dy=1, **self.kwargs)
        grad_ad = -dlS_dlP_T / dlS_dlT_P

        dlRho_dlT_P = tpt_zi.interp_pt_logRho_z(logT, logP, dx=1, **self.kwargs)
        dlRho_dlP_T = tpt_zi.interp_pt_logRho_z(logT, logP, dy=1, **self.kwargs)
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        input_shape = logT.shape
        res_z = get_zeros(input_shape=input_shape, num_vals=self.num_vals_for_evaluate)
        res_z[self.i_logRho] = logRho
        res_z[self.i_logS] = logS
        res_z[self.i_dlS_dlP] = dlS_dlP_T
        res_z[self.i_dlS_dlT] = dlS_dlT_P
        res_z[self.i_grad_ad] = grad_ad
        res_z[self.i_chiRho] = chiRho
        res_z[self.i_chiT] = chiT
        return res_z

    def evaluate(
        self,
        logT: ArrayLike,
        logP: ArrayLike,
        X: ArrayLike = 0,
        Z1: ArrayLike = 0,
        Z2: ArrayLike = 0,
        Z3: ArrayLike = 0,
    ) -> NDArray:
        """Calculates the equation of state output for the mixture.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logP (ArrayLike): log10 of the pressure.
            X (ArrayLike, optional): hydrogen mass-fraction. Defaults to 0.
            Z1 (ArrayLike, optional): first heavy-element mass-fraction.
                Defaults to 0.
            Z2 (ArrayLike, optional): second heavy-element mass-fraction.
                Defaults to 0.
            Z3 (ArrayLike, optional): third heavy-element mass-fraction.
                Defaults to 0.

        Returns:
            NDArray: logRho, logS, grad_ad, chiRho, chiT and c_sound.
        """
        # check the input and make sure everything
        # has the same shape
        logT, logP = self.__check_PT(logT, logP)
        X, Y, Z1, Z2, Z3 = self.__check_composition(X, Z1, Z2, Z3)
        if logT.ndim > X.ndim:
            X = X * np.ones_like(logT)
            Y = Y * np.ones_like(logT)
            Z1 = Z1 * np.ones_like(logT)
            Z2 = Z2 * np.ones_like(logT)
            Z3 = Z3 * np.ones_like(logT)
        elif logT.ndim < X.ndim:
            logT = logT * np.ones_like(X)
            logP = logP * np.ones_like(X)
        self.input_shape = logT.shape
        self.input_ndim = len(logT.shape)

        X_one = np.isclose(X, 1, atol=eps1)
        if np.all(np.isclose(X, 0, atol=eps1)):
            res_x = get_zeros(
                input_shape=self.input_shape,
                num_vals=self.num_vals_for_evaluate,
                empty=True,
            )
            res_x.fill(tiny_val)
        else:
            if np.all(X_one) or not self.include_hhe_interactions:
                res_x = self.__evaluate_x(logT, logP)
            elif np.any(X_one) and self.include_hhe_interactions:
                i_xeff = X < 1
                i_x = ~i_xeff
                logT_x = logT[i_x]
                logP_x = logP[i_x]
                logT_xeff = logT[i_xeff]
                logP_xeff = logP[i_xeff]
                res_x = get_zeros(
                    input_shape=self.input_shape, num_vals=self.num_vals_for_evaluate
                )
                res_x[:, i_x] = self.__evaluate_x(logT_x, logP_x)
                res_x[:, i_xeff] = self.__evaluate_x_eff(logT_xeff, logP_xeff)
            else:
                res_x = self.__evaluate_x_eff(logT, logP)

        if np.all(np.isclose(Y, 0, atol=eps1)):
            res_y = get_zeros(
                input_shape=self.input_shape,
                num_vals=self.num_vals_for_evaluate,
                empty=True,
            )
            res_y.fill(tiny_val)
        else:
            res_y = self.__evaluate_y(logT, logP)

        if np.all(np.isclose(Z1, 0, atol=eps1)):
            res_z1 = get_zeros(
                input_shape=self.input_shape,
                num_vals=self.num_vals_for_evaluate,
                empty=True,
            )
            res_z1.fill(tiny_val)
        else:
            res_z1 = self.__evaluate_z(0, logT, logP)

        if np.all(np.isclose(Z2, 0, atol=eps1)):
            res_z2 = get_zeros(
                input_shape=self.input_shape,
                num_vals=self.num_vals_for_evaluate,
                empty=True,
            )
            res_z2.fill(tiny_val)
        else:
            res_z2 = self.__evaluate_z(1, logT, logP)

        if np.all(np.isclose(Z3, 0, atol=eps1)):
            res_z3 = get_zeros(
                input_shape=self.input_shape,
                num_vals=self.num_vals_for_evaluate,
                empty=True,
            )
            res_z3.fill(tiny_val)
        else:
            res_z3 = self.__evaluate_z(2, logT, logP)

        logRho_x, logS_x, dlS_dlP_x, dlS_dlT_x, _ = self.__unpack(res_x)
        logRho_y, logS_y, dlS_dlP_y, dlS_dlT_y, _ = self.__unpack(res_y)
        logRho_z1, logS_z1, dlS_dlP_z1, dlS_dlT_z1, _ = self.__unpack(res_z1)
        logRho_z2, logS_z2, dlS_dlP_z2, dlS_dlT_z2, _ = self.__unpack(res_z2)
        logRho_z3, logS_z3, dlS_dlP_z3, dlS_dlT_z3, _ = self.__unpack(res_z3)

        rho_x = 10**logRho_x
        rho_y = 10**logRho_y
        rho_z1 = 10**logRho_z1
        rho_z2 = 10**logRho_z2
        rho_z3 = 10**logRho_z3

        rho_inv = X / rho_x + Y / rho_y + Z1 / rho_z1 + Z2 / rho_z2 + Z3 / rho_z3
        logRho = -np.log10(rho_inv)

        S_x = 10**logS_x
        S_y = 10**logS_y
        S_z1 = 10**logS_z1
        S_z2 = 10**logS_z2
        S_z3 = 10**logS_z3
        S = X * S_x + Y * S_y + Z1 * S_z1 + Z2 * S_z2 + Z3 * S_z3
        Z_tot = Z1 + Z2 + Z3
        A_z_mean = (
            Z1 * self.A1 + Z2 * self.A2 + Z3 * self.A3
        ) / Z_tot  # mean atomic weight of the three heavy elements
        S = S + get_mixing_entropy(Y=Y, Z=Z_tot, A_z=A_z_mean)
        logS = np.log10(S)

        dlS_dlP = (
            X * S_x * dlS_dlP_x
            + Y * S_y * dlS_dlP_y
            + Z1 * S_z1 * dlS_dlP_z1
            + Z2 * S_z2 * dlS_dlP_z2
            + Z3 * S_z3 * dlS_dlP_z3
        ) / S
        dlS_dlT = (
            X * S_x * dlS_dlT_x
            + Y * S_y * dlS_dlT_y
            + Z1 * S_z1 * dlS_dlT_z1
            + Z2 * S_z2 * dlS_dlT_z2
            + Z3 * S_z3 * dlS_dlT_z3
        ) / S
        grad_ad = -dlS_dlP / dlS_dlT

        shape = (5, 2) + logT.shape if self.input_ndim > 0 else (5, 2)
        fac = np.zeros(shape)
        fac[0, 0] = X / rho_x / res_x[self.i_chiRho]
        fac[0, 1] = -fac[0, 0] * res_x[self.i_chiT]
        fac[1, 0] = Y / rho_y / res_y[self.i_chiRho]
        fac[1, 1] = -fac[1, 0] * res_y[self.i_chiT]
        fac[2, 0] = Z1 / rho_z1 / res_z1[self.i_chiRho]
        fac[2, 1] = -fac[2, 0] * res_z1[self.i_chiT]
        fac[3, 0] = Z2 / rho_z2 / res_z2[self.i_chiRho]
        fac[3, 1] = -fac[3, 0] * res_z2[self.i_chiT]
        fac[4, 0] = Z3 / rho_z3 / res_z3[self.i_chiRho]
        fac[4, 1] = -fac[4, 0] * res_z3[self.i_chiT]

        dlRho_dlP_T = (1 / rho_inv) * (
            fac[0, 0] + fac[1, 0] + fac[2, 0] + fac[3, 0] + fac[4, 0]
        )
        dlRho_dlT_P = (1 / rho_inv) * (
            fac[0, 1] + fac[1, 1] + fac[2, 1] + fac[3, 1] + fac[4, 1]
        )
        chiRho = 1 / dlRho_dlP_T
        chiT = -dlRho_dlT_P / dlRho_dlP_T

        if self.limit_bad_values:
            check_logS = np.isnan(logS)
            check_grad_ad = np.isnan(grad_ad)
            if np.any(check_logS):
                if self.input_ndim > 0:
                    logS[check_logS] = -10
                else:
                    logS = -10
            if np.any(check_grad_ad):
                if self.input_ndim > 0:
                    grad_ad[check_grad_ad] = self.lower_grad_ad
                else:
                    grad_ad = self.lower_grad_ad
            logS = np.clip(a=logS, a_min=-10, a_max=12)
            grad_ad = np.clip(
                a=grad_ad, a_min=self.lower_grad_ad, a_max=self.upper_grad_ad
            )
            chiRho = np.clip(a=chiRho, a_min=self.lower_chiRho, a_max=self.upper_chiRho)
            chiT = np.clip(a=chiT, a_min=self.lower_chiT, a_max=self.upper_chiT)

        gamma1 = chiRho / (1 - chiT * grad_ad)
        c_sound = np.sqrt(10**logP / 10**logRho * gamma1)

        res = get_zeros(input_shape=self.input_shape, num_vals=self.num_vals_for_return)
        res[0] = logRho
        res[1] = logS
        res[2] = grad_ad
        res[3] = chiRho
        res[4] = chiT
        res[5] = c_sound
        return res
