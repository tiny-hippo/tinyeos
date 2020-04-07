import os
import numpy as np
from tinyeos.tableloader import TableLoader
from tinyeos.interpolantsbuilder import InterpolantsBuilder
from tinyeos.support import get_eta, ideal_mixing_law, check_composition
from tinyeos.definitions import *
from scipy.optimize import root_scalar
from pathlib import Path


class TinyDT(InterpolantsBuilder):
    """ Temperature-density equation of state for a mixture of hydrogen,
    helium and a heavy element.

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

    Attributes:
        # to-do: list attributes
    """
    def __init__(self,
                 which_heavy="water",
                 which_hhe="cms",
                 build_interpolants=False) -> None:
        """ __init__ method. Defines parameters and either loads or
        builds the interpolants.

        Args:
            which_heavy (str, optional): Which heavy-element equation of state
            to use. Defaults to "water". Options are "water", "rock",
            "mixture", "aqua" or "iron".
            which_hhe (str, optional): Which hydrogen-helium equation of state
            to use. Defaults to "cms". Options are "cms" or "scvh".
            build_interpolants (bool, optional): Whether to build interpolants.
            Defaults to False.

        Raises:
            NotImplementedError: Raised if which_heavy or which_hhe choices
            are unavailable.
        """

        if build_interpolants:
            super().__init__()

        self.logRho_max = logRho_max
        self.logRho_min = logRho_min
        self.logT_max = logT_max
        self.logT_min = logT_min
        self.eps = 1e-3

        if which_hhe == "scvh":
            self.logRho_max = 2
            self.logRho_min = -12

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

        self.kwargs = {"grid": False}
        self.cache_path = Path(__file__).parent / "data/interpolants"
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

        self.interpPT_x = self.load_interp("interpPT_x_" + which_hhe + ".npy")
        self.interpPT_y = self.load_interp("interpPT_y_" + which_hhe + ".npy")
        self.interpPT_z = self.load_interp("interpPT_z_" + which_heavy +
                                           ".npy")

        self.interpDT_x = self.load_interp("interpDT_x_" + which_hhe + ".npy")
        self.interpDT_y = self.load_interp("interpDT_y_" + which_hhe + ".npy")
        self.interpDT_z = self.load_interp("interpDT_z_" + which_heavy +
                                           ".npy")

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
        self.interpPT_logRho_y = self.interpPT_y[0]
        self.interpPT_logRho_z = self.interpPT_z[0]
        self.interpPT_logS_z = self.interpPT_z[1]
        self.interpPT_logU_z = self.interpPT_z[2]

    def load_interp(self, filename) -> object:
        """ Loads the interpolant from the disk.

        Args:
            filename (str): Name of the interpolant cache file.

        Raises:
            FileNotFoundError: Raised if the interpolant was not found.

        Returns:
            object: Bivariate spline loaded from the cache file.
        """

        src = os.path.join(self.cache_path, filename)
        if not os.path.isfile(src):
            raise FileNotFoundError("Missing interpolant cache" + src)
        return np.load(src, allow_pickle=True)

    def check_DT(self, logT, logRho) -> None:
        """ Makes sure that input temperature and density
        are within equation of state limits.

        Args:
            logT (float or ndarray): Log10 of the temperature.
            logRho (float or ndarray): Log10 of the density.
        """

        assert (np.all(logT >= self.logT_min)
                and np.all(logT <= self.logT_max))
        assert (np.all(logRho >= self.logRho_min)
                and np.all(logRho <= self.logRho_max))

    def ideal_mixture(self, logT, logRho, X, Y, Z, debug=False) -> tuple:
        """ Calculates the individual densities of the elements in the mixture.

        Args:
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.
            X (float): Hydrogen mass-fraction
            Y (float): Helium mass-fraction.
            Z (float): Heavy-element mass-fraction.
            debug (bool, optional): Enables additional output.
            Defaults to False.

        Returns:
            tuple: Tuple consisting of convergence information, individual
            densities and gas pressure.
        """

        self.check_DT(logT, logRho)

        if Z == 1:
            conv = True
            logP = self.interpDT_logP_z(logT, logRho, **self.kwargs)
            logRho_x = self.logRho_min
            logRho_y = self.logRho_min
            logRho_z = logRho

        elif X == 1:
            conv = True
            logP = self.interpDT_logP_x(logT, logRho, **self.kwargs)
            logRho_x = logRho
            logRho_y = self.logRho_min
            logRho_z = self.logRho_min

        elif Y == 1:
            conv = True
            logP = self.interpDT_logP_y(logT, logRho, **self.kwargs)
            logRho_x = self.logRho_min
            logRho_y = logRho
            logRho_z = self.logRho_min

        elif Z > 0:
            have_bracket = True
            x0 = self.logRho_max
            x1 = self.logRho_min
            # have_bracket, x0, x1 = self.get_bracket(logT, logRho, X, Y, Z)

            if have_bracket:
                # found bracket
                try:
                    sol = root_scalar(self.optimize_1d,
                                      args=(logT, logRho, X, Y, Z),
                                      method="brentq",
                                      bracket=[x0, x1])
                    conv = sol.converged
                    logRho_z = sol.root
                except ValueError:
                    conv = False
                    logRho_z = self.logRho_min
            else:
                # no bracket found
                # x0 = logRho + np.log10(Z)
                # x1 = x0 - 0.05
                # sol = root_scalar(self.optimize_1d,
                #                   args=(logT, logRho, X, Y, Z),
                #                   method="secant",
                #                   x0=x0,
                #                   x1=x1)
                conv = False
                logRho_z = self.logRho_min

            logP = self.interpDT_logP_z(logT, logRho_z, **self.kwargs)
            if X > 0 and conv:
                logRho_x = self.interpPT_logRho_x(logT, logP, **self.kwargs)
            else:
                logRho_x = self.logRho_min
            if Y > 0 and conv:
                logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)
            else:
                logRho_y = self.logRho_min

        elif X > 0:
            # have_bracket, x0, x1 = self.get_xy_bracket(logT, logRho, X, Y)
            have_bracket = True
            x0 = self.logRho_min
            x1 = self.logRho_max
            if have_bracket:
                try:
                    sol = root_scalar(self.optimize_1d,
                                      args=(logT, logRho, X, Y, Z),
                                      method="brentq",
                                      bracket=[x0, x1])
                    conv = sol.converged
                    logRho_x = sol.root
                except ValueError:
                    conv = False
                    logRho_x = self.logRho_min

            else:
                conv = False
                logRho_x = self.logRho_min
            #     # no bracket found
            #     x0 = logRho + np.log10(X)
            #     x1 = x0 - 0.05
            #     sol = root_scalar(self.optimize_1d,
            #                       args=(logT, logRho, X, Y, Z),
            #                       method="secant",
            #                       x0=x0,
            #                       x1=x1)

            logP = self.interpDT_logP_x(logT, logRho_x, **self.kwargs)
            logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)
            logRho_z = self.logRho_min

        if (debug):
            res = self.optimize_1d(logRho_z, logRho, logT, X, Y, Z)
            logP_x = self.interpDT_logP_x(logT, logRho_x, **self.kwargs)
            logP_y = self.interpDT_logP_y(logT, logRho_y, **self.kwargs)

            print(sol)
            print("Residual: {:.2E}".format(res))
            print("logP: {:.3f} {:.3f} {:.3f}".format(logP, logP_x, logP_y))

        if not conv:
            pass
        elif not (self.logRho_min <= logRho_z <= self.logRho_max):
            conv = False
        elif not (self.logRho_min <= logRho_x <= self.logRho_max):
            conv = False
        elif not (self.logRho_min <= logRho_y <= self.logRho_max):
            conv = False

        return (conv, logRho_x, logRho_y, logRho_z, logP)

    def optimize_1d(self, xhat, logT, logRho, X, Y, Z) -> float:
        """ Defines the root finding problem to find the
        individual densities. The root will be either the
        density of the heavy element (if Z > 0) or the
        hydrogen density.

        Args:
            xhat (float): Log10 of either the heavy-element
            or hydrogen density.
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.
            X (float): Hydrogen mass-fraction.
            Y (float): Helium mass-fraction.
            Z (float): Heavy-element mass-fraction.

        Returns:
            float: Residual for the root finding algorithm.
        """

        rho = 10**logRho
        if Z > 0:
            logP = self.interpDT_logP_z(logT, xhat, **self.kwargs)
            rho_z = 10**xhat
        elif (X > 0):
            logP = self.interpDT_logP_x(logT, xhat, **self.kwargs)
            rho_x = 10**xhat
            rho_z = 0

        if X == 0:
            rho_x = 0
        else:
            logRho_x = self.interpPT_logRho_x(logT, logP, **self.kwargs)
            rho_x = 10**logRho_x
        if Y == 0:
            rho_y = 0
        else:
            logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)
            rho_y = 10**logRho_y

        f = np.log10(1 / rho) - np.log10(
            ideal_mixing_law(rho_x, rho_y, rho_z, X, Y, Z))
        return f

    def get_bracket(self, logT, logRho, X, Y, Z) -> tuple:
        """ Finds the bracket in the heavy-element density.

        Args:
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.
            X (float): Hydrogen mass-fraction.
            Y (float): Helium mass-fraction.
            Z (float): Heavy-element mass-fraction.

        Returns:
            tuple: Tuple consisting of convergence information and the bracket.
        """

        rho0 = 10**logRho
        rho_z = Z * rho0
        logRho_z = np.max([self.logRho_min, np.log10(rho_z)])

        logP = self.interpDT_logP_z(logT, logRho_z, **self.kwargs)
        logRho_x = self.interpPT_logRho_x(logT, logP, **self.kwargs)
        logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)

        rho_x = 10**logRho_x
        rho_y = 10**logRho_y
        rho = 1 / ideal_mixing_law(rho_x, rho_y, rho_z, X, Y, Z)

        if rho < rho0:
            logRho_z_low = logRho_z
        elif rho > rho0:
            print("rho > rho0")

        logRho_z_max = self.logRho_max
        logRho_z = logRho_z_low + 0.05
        rho_z = 10**logRho_z

        max_iter = 5000
        have_bracket = False
        for i in range(max_iter):
            rho_old = rho
            logP = self.interpDT_logP_z(logT, logRho_z, **self.kwargs)
            logRho_x = self.interpPT_logRho_x(logT, logP, **self.kwargs)
            logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)

            rho_x = 10**logRho_x
            rho_y = 10**logRho_y
            rho = 1 / ideal_mixing_law(rho_x, rho_y, rho_z, X, Y, Z)

            if rho < rho0:
                logRho_z_low = logRho_z
                logRho_z = logRho_z_low + 0.05
                if logRho_z_max - logRho_z < 1e-6:
                    have_bracket = False
                    break
                rho_z = 10**logRho_z
            elif rho > rho0:
                logRho_z_high = logRho_z
                have_bracket = True
                break

        x0 = logRho_z_low
        if have_bracket:
            x1 = logRho_z_high
        else:
            x0 = self.logRho_min
            x1 = self.logRho_max

        return (have_bracket, x0, x1)

    def get_xy_bracket(self, logT, logRho, X, Y) -> tuple:
        """ Finds the bracket in the hydrogen density.

        Args:
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.
            X (float): Hydrogen mass-fraction.
            Y (float): Helium mass-fraction.

        Returns:
            tuple: Tuple consisting of convergence information and the bracket.
        """

        rho0 = 10**logRho
        rho_x = X * rho0
        logRho_x = np.max([self.logRho_min, np.log10(rho_x)])

        logP = self.interpDT_logP_x(logT, logRho_x, **self.kwargs)
        logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)

        rho_y = 10**logRho_y
        rho = 1 / ideal_mixing_law(rho_x, rho_y, 0, X, Y, 0)

        if rho < rho0:
            logRho_x_low = logRho_x
        elif rho > rho0:
            print("rho > rho0")

        logRho_x_max = self.logRho_max
        logRho_x = logRho_x_low + 0.05
        rho_x = 10**logRho_x

        max_iter = 5000
        have_bracket = False
        for i in range(max_iter):
            rho_old = rho
            logP = self.interpDT_logP_x(logT, logRho_x, **self.kwargs)
            logRho_y = self.interpPT_logRho_y(logT, logP, **self.kwargs)

            rho_x = 10**logRho_x
            rho_y = 10**logRho_y
            rho = 1 / ideal_mixing_law(rho_x, rho_y, 0, X, Y, 0)

            if rho < rho0:
                logRho_x_low = logRho_x
                logRho_x = logRho_x_low + 0.05
                if logRho_x_max - logRho_x < 1e-6:
                    have_bracket = False
                    break
                rho_x = 10**logRho_x
            elif rho > rho0:
                logRho_x_high = logRho_x
                have_bracket = True
                break

        x0 = logRho_x_low
        if have_bracket:
            x1 = logRho_x_high
        else:
            x0 = self.logRho_min
            x1 = self.logRho_max

        return (have_bracket, x0, x1)

    def evaluate_x(self, logT, logRho) -> np.ndarray:
        """ Calculates equation of state output for hydrogen.

        Args:
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.

        Returns:
            np.ndarray: Equation of state output.
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

        res_x = np.zeros(self.num_vals)
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

    def evaluate_y(self, logT, logRho) -> np.ndarray:
        """ Calculates equation of state output for hydrogen.

        Args:
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.

        Returns:
            np.ndarray: Equation of state output.
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

        res_y = np.zeros(self.num_vals)
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

    def evaluate_z(self, logT, logRho) -> np.ndarray:
        """ Calculates equation of state output for the heavy element.

        Args:
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.

        Returns:
            np.ndarray: Equation of state output.
        """

        logP = self.interpDT_logP_z(logT, logRho, **self.kwargs)
        logS = self.interpDT_logS_z(logT, logRho, **self.kwargs)
        logU = self.interpDT_logU_z(logT, logRho, **self.kwargs)

        chiRho = self.interpDT_logP_z(logT, logRho, dy=1, **self.kwargs)
        chiT = self.interpDT_logP_z(logT, logRho, dx=1, **self.kwargs)

        dlS_dlT = self.interpDT_logS_z(logT, logRho, dx=1, **self.kwargs)
        dlS_dlRho = self.interpDT_logS_z(logT, logRho, dy=1, **self.kwargs)

        if self.heavy_element == "aqua":
            grad_ad = self.interpDT_grad_ad_z(logT, logRho, **self.kwargs)
        else:
            grad_ad = 1 / (chiT - dlS_dlT * chiRho / dlS_dlRho)

        res_z = np.zeros(self.num_vals)
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

    def evaluate(self,
                 logT,
                 logRho,
                 X,
                 Z,
                 fix_phase_transition=False,
                 debug=False) -> np.ndarray:
        """ Calculates the equation of state output for the mixture.

        Args:
            logT (float): Log10 of the temperature.
            logRho (float): Log10 of the density.
            X (float): Hydrogen mass-fraction.
            Z (float): Heavy-element mass-fraction.
            fix_phase_transition (bool, optional): Whether to attempt to fix
            phase transitions. Defaults to False.
            debug (bool, optional): Whether to enable additional output.
            Defaults to False.

        Returns:
            np.ndarray: Equation of state output. The index of the individual
            quantities is defined in the __init__ method. If the root finding
            algorithm failed, the output will be filled with negative ones.
        """

        self.check_DT(logT, logRho)

        X, Y, Z = check_composition(X, Z)

        xhat = self.ideal_mixture(logT, logRho, X, Y, Z)
        if not xhat[0]:
            if debug:
                print("Failed in density root find")
            res = -np.ones(self.num_vals)
            return res

        logRho_x = xhat[1]
        logRho_y = xhat[2]
        logRho_z = xhat[3]
        logP = xhat[4]

        T = 10**logT
        P = 10**logP
        rho = 10**logRho
        rho_x = 10**logRho_x
        rho_y = 10**logRho_y
        rho_z = 10**logRho_z

        if X > 0:
            res_x = self.evaluate_x(logT, logRho_x)
        else:
            res_x = np.zeros(self.num_vals)
        if Y > 0:
            res_y = self.evaluate_y(logT, logRho_y)
        else:
            res_y = np.zeros(self.num_vals)
        if Z > 0:
            res_z = self.evaluate_z(logT, logRho_z)
        else:
            res_z = np.zeros(self.num_vals)

        # check if we ran into a phase transition
        # if yes, re-do ideal mixing calculation while ignoring
        # the heavy-element contribution
        if fix_phase_transition:
            if res_z[self.i_chiRho] <= 0 or res_z[self.i_chiT] <= 0 and Z < 1:
                if debug:
                    print(f"Phase transition: {logT:.2f} {logRho:.2f}")
                # re-define mass fractions yielding the same X/Y ratio
                XY_ratio = X / Y
                Y = 1 / (XY_ratio + 1)
                X = Y * XY_ratio
                # re-do ideal mixing calculation with Z = 0
                Z = 0
                xhat = self.ideal_mixture(logT, logRho, X, Y, Z)
                if not xhat[0]:
                    if debug:
                        print("Failed in density root find for logT, logRho =",
                              f"{logT:.2f}, {logRho:.2f}",
                              f"and X, Z = {X:.2f}, {Y:.2f}")
                        res = -np.ones(self.num_vals)
                        return res

                logRho_x = xhat[1]
                logRho_y = xhat[2]
                logRho_z = xhat[3]
                logP = xhat[4]

                T = 10**logT
                P = 10**logP
                rho = 10**logRho
                rho_x = 10**logRho_x
                rho_y = 10**logRho_y
                rho_z = 10**logRho_z

        if X > 0:
            res_x = self.evaluate_x(logT, logRho_x)
        else:
            res_x = np.zeros(self.num_vals)
        if Y > 0:
            res_y = self.evaluate_y(logT, logRho_y)
        else:
            res_y = np.zeros(self.num_vals)
        if Z > 0:
            res_z = self.evaluate_z(logT, logRho_z)
        else:
            res_z = np.zeros(self.num_vals)

        if X == 1:
            logP = res_x[self.i_logP]
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
                dlS_dlT_P = self.interpDT_dlS_dlT_P_x(logT, logRho,
                                                      **self.kwargs)
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
            logP = res_y[self.i_logP]
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
                dlS_dlT_P = self.interpDT_dlS_dlT_P_y(logT, logRho,
                                                      **self.kwargs)
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
            logP = res_z[self.i_logP]
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
                dlS_dlT_P = self.interpPT_logS_z(logT,
                                                 logP,
                                                 dx=1,
                                                 **self.kwargs)
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

            dlS_dlP_T_y = self.interpDT_dlS_dlP_T_y(logT, logRho_y,
                                                    **self.kwargs)
            dlS_dlT_P_y = self.interpDT_dlS_dlT_P_y(logT, logRho_y,
                                                    **self.kwargs)
            dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
            dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

            dlS_dlP_T = (Y * S_y * dlS_dlP_T_y + Z * S_z * dlS_dlP_T_z) / S
            dlS_dlT_P = (Y * S_y * dlS_dlT_P_y + Z * S_z * dlS_dlT_P_z) / S

            dlRho_dlP_T = rho * (Y / rho_y / res_y[self.i_chiRho] +
                                 Z / rho_z / res_z[self.i_chiRho])
            dlRho_dlT_P = rho * (Y / rho_y *
                                 (-res_y[self.i_chiT] / res_y[self.i_chiRho]) +
                                 Z / rho_z *
                                 (-res_z[self.i_chiT] / res_z[self.i_chiRho]))

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

            dlS_dlP_T_x = self.interpDT_dlS_dlP_T_x(logT, logRho_x,
                                                    **self.kwargs)
            dlS_dlT_P_x = self.interpDT_dlS_dlT_P_x(logT, logRho_x,
                                                    **self.kwargs)
            dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
            dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

            dlS_dlP_T = (X * S_x * dlS_dlP_T_x + Z * S_z * dlS_dlP_T_z) / S
            dlS_dlT_P = (X * S_x * dlS_dlT_P_x + Z * S_z * dlS_dlT_P_z) / S

            dlRho_dlP_T = rho * (X / rho_x / res_x[self.i_chiRho] +
                                 Z / rho_z / res_z[self.i_chiRho])
            dlRho_dlT_P = rho * (X / rho_x *
                                 (-res_x[self.i_chiT] / res_x[self.i_chiRho]) +
                                 Z / rho_z *
                                 (-res_z[self.i_chiT] / res_z[self.i_chiRho]))

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

            dlS_dlP_T_x = self.interpDT_dlS_dlP_T_x(logT, logRho_x,
                                                    **self.kwargs)
            dlS_dlT_P_x = self.interpDT_dlS_dlT_P_x(logT, logRho_x,
                                                    **self.kwargs)
            dlS_dlP_T_y = self.interpDT_dlS_dlP_T_y(logT, logRho_y,
                                                    **self.kwargs)
            dlS_dlT_P_y = self.interpDT_dlS_dlT_P_y(logT, logRho_y,
                                                    **self.kwargs)

            dlS_dlP_T = (X * S_x * dlS_dlP_T_x + Y * S_y * dlS_dlP_T_y) / S
            dlS_dlT_P = (X * S_x * dlS_dlT_P_x + Y * S_y * dlS_dlT_P_y) / S

            dlRho_dlP_T = rho * (X / rho_x / res_x[self.i_chiRho] +
                                 Y / rho_y / res_y[self.i_chiRho])
            dlRho_dlT_P = rho * (X / rho_x *
                                 (-res_x[self.i_chiT] / res_x[self.i_chiRho]) +
                                 Y / rho_y *
                                 (-res_y[self.i_chiT] / res_y[self.i_chiRho]))

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

            lfe = np.log10(X * 10**res_x[self.i_lfe] +
                           Y * 10**res_y[self.i_lfe])
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

            dlS_dlP_T_x = self.interpDT_dlS_dlP_T_x(logT, logRho_x,
                                                    **self.kwargs)
            dlS_dlT_P_x = self.interpDT_dlS_dlT_P_x(logT, logRho_x,
                                                    **self.kwargs)
            dlS_dlP_T_y = self.interpDT_dlS_dlP_T_y(logT, logRho_y,
                                                    **self.kwargs)
            dlS_dlT_P_y = self.interpDT_dlS_dlT_P_y(logT, logRho_y,
                                                    **self.kwargs)
            dlS_dlP_T_z = self.interpPT_logS_z(logT, logP, dy=1, **self.kwargs)
            dlS_dlT_P_z = self.interpPT_logS_z(logT, logP, dx=1, **self.kwargs)

            dlS_dlP_T = (X * S_x * dlS_dlP_T_x + Y * S_y * dlS_dlP_T_y +
                         Z * S_z * dlS_dlP_T_z) / S
            dlS_dlT_P = (X * S_x * dlS_dlT_P_x + Y * S_y * dlS_dlT_P_y +
                         Z * S_z * dlS_dlT_P_z) / S

            dlRho_dlP_T = rho * (X / rho_x / res_x[self.i_chiRho] +
                                 Y / rho_y / res_y[self.i_chiRho] +
                                 Z / rho_z / res_z[self.i_chiRho])
            dlRho_dlT_P = rho * (
                X / rho_x *
                (-res_x[self.i_chiT] / res_x[self.i_chiRho]) + Y / rho_y *
                (-res_y[self.i_chiT] / res_y[self.i_chiRho]) + Z / rho_z *
                (-res_z[self.i_chiT] / res_z[self.i_chiRho]))

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
            # c_sound = np.sqrt(P / rho * gamma1)
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
            lfe = np.log10(X * 10**res_x[self.i_lfe] +
                           Y * 10**res_y[self.i_lfe])
            eta = get_eta(logT, logRho, lfe)

            # crude estimate for mean molecular weight of the heavy element
            mu = 1 / (X / res_x[self.i_mu] + Y / res_y[self.i_mu] +
                      Z / res_z[self.i_mu])

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

        return res
