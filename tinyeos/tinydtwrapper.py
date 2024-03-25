from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

from tinyeos.definitions import i_logRho, i_logT, num_vals
from tinyeos.tinypteos import TinyPT


class TinyDTWrapper:
    def __init__(
        self,
        which_heavy: str = "h2o",
        which_hhe: str = "cms",
        *,
        include_hhe_interactions: bool = False,
        use_smoothed_xy_tables: bool = False,
        use_smoothed_z_tables: bool = False,
        build_interpolants: bool = False,
    ) -> None:
        """__init__ method.

        Defines parameters and either loads or
        builds the interpolants.

        Args:
        ----
            which_heavy (str, optional): heavy-element equation of state
                to use. Defaults to "h2o". Options are "h2o", "aqua", "sio2",
                "mixture", "fe" or "co".
            which_hhe (str, optional): hydrogen-helium equation of state
                to use. Defaults to "cms". Options are "cms" or "scvh".
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
            which_heavy=which_heavy,
            which_hhe=which_hhe,
            include_hhe_interactions=include_hhe_interactions,
            use_smoothed_xy_tables=use_smoothed_xy_tables,
            use_smoothed_z_tables=use_smoothed_z_tables,
            build_interpolants=build_interpolants,
        )

    def __call__(self, logT: float, logRho: float, X: float, Z: float) -> NDArray:
        """__call__ method acting as convenience wrapper for the evaluate method.

        Calculates the equation of state output for the mixture.
        Only scalar inputs are supported. If the root find fails,
        the results are set to nans.

        Args:
        ----
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            X (float): hydrogen mass fraction.
            Z (float): heavy-element mass fraction.

        Returns:
        -------
            NDArray: equation of state output. The indices of the
                individual quantities are defined in definitions.py.

        """
        return self.evaluate(logT=logT, logRho=logRho, X=X, Z=Z)

    def __helper(
        self,
        logP: float,
        logT: float,
        logRho: float,
        X: float,
        Y: float,
        Z: float,
    ) -> float:
        """Helper function for the root finding.

        Calls the pressure-temperature equation of state and returns
        the difference between the input and the calculated density.

        Args:
        ----
            logP (float): log10 of the pressure.
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            X (float): hydrogen mass fraction.
            Y (float): helium mass fraction.
            Z (float): heavy-element mass fraction.

        Returns:
        -------
            float: difference of the input and calculated density.

        """
        logRho_iml = self.tpt._TinyPT__ideal_mixture(
            logT=logT, logP=logP, X=X, Y=Y, Z=Z
        )
        return logRho - logRho_iml

    def __root_finder(
        self,
        logT: float,
        logRho: float,
        X: float,
        Y: float,
        Z: float,
    ) -> Tuple[bool, float]:
        """Root finding function.

        Uses the pressure-temperature
        equation of state to find the pressure corresponding
        to the input density.

        Args:
        ----
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            X (float): hydrogen mass fraction.
            Y (float): helium mass fraction.
            Z (float): heavy-element mass fraction.

        Returns:
        -------
            Tuple[bool, float]: root finding result.

        """
        logP0 = self.tpt.logP_min
        logP1 = self.tpt.logP_max
        f1 = self.__helper(logP=logP0, logT=logT, logRho=logRho, X=X, Y=Y, Z=Z)
        f2 = self.__helper(logP=logP1, logT=logT, logRho=logRho, X=X, Y=Y, Z=Z)
        if np.sign(f1) == np.sign(f2):
            converged = False
            root = np.nan
        else:
            sol = root_scalar(
                f=self.__helper,
                args=(logT, logRho, X, Y, Z),
                method="brentq",
                bracket=[logP0, logP1],
            )
            converged = sol.converged
            root = sol.root
        return (converged, root)

    def evaluate(self, logT: float, logRho: float, X: float, Z: float) -> NDArray:
        """Calculate the equation of state output for the mixture.

        Only scalar inputs are supported. If the root find fails,
        the results are set to nans.

        Args:
        ----
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            X (float): hydrogen mass fraction.
            Z (float): heavy-element mass fraction.

        Returns:
        -------
            NDArray: equation of state output. The indices of the
                individual quantities are defined in definitions.py.

        """
        _, _, X, Y, Z, _ = self.tpt._TinyPT__prepare(
            logT=logT, logP=6, X=X, Z=Z
        )
        converged, logP = self.__root_finder(logT=logT, logRho=logRho, X=X, Y=Y, Z=Z)
        if not converged:
            res = np.zeros(num_vals)
            res[i_logT] = logT
            res[i_logRho] = logRho
            res[i_logRho + 1 :] = np.nan
        else:
            res = self.tpt.evaluate(logT=logT, logP=logP, X=X, Z=Z)
        return res
