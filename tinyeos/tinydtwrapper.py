from typing import tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root_scalar
from scipy.optimize.elementwise import bracket_root, find_root

from tinyeos.definitions import i_logRho, i_logT
from tinyeos.tinypteos import TinyPT


class TinyDTWrapper:
    def __init__(
        self,
        which_heavy: str = "h2o",
        which_hhe: str = "cms",
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

        self.logRho_max = self.tpt.logRho_max
        self.logRho_min = self.tpt.logRho_min
        self.logP_max = self.tpt.logP_max
        self.logP_min = self.tpt.logP_min
        self.logT_max = self.tpt.logT_max
        self.logT_min = self.tpt.logT_min

    def __call__(
        self, logT: ArrayLike, logRho: ArrayLike, X: ArrayLike, Z: ArrayLike
    ) -> NDArray:
        """__call__ method acting as convenience wrapper for the evaluate method.

        Calculates the equation of state output for the mixture.
        Only scalar inputs are supported. If the root find fails,
        the results are set to nans.

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
        """Root finding function.

        Uses the pressure-temperature
        equation of state to find the pressure corresponding
        to the input density.

        Args:
        ----
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            X (ArrayLike): hydrogen mass fraction.
            Y (ArrayLike): helium mass fraction.
            Z (ArrayLike): heavy-element mass fraction.

        Returns:
        -------
            Tuple[bool, ArrayLike]: root finding result.

        """
        if logT.ndim == 0:
            f1 = self.__helper(
                logP=self.tpt.logP_min, logT=logT, logRho=logRho, X=X, Y=Y, Z=Z
            )
            f2 = self.__helper(
                logP=self.tpt.logP_max, logT=logT, logRho=logRho, X=X, Y=Y, Z=Z
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

        Only scalar inputs are supported. If the root find fails,
        the results are set to nans.

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
                res[i_logT] = logT
                res[i_logRho] = logRho
                res[i_logRho + 1 :] = np.nan
        else:
            if np.all(success):
                res[...] = self.tpt.evaluate(logT=logT, logP=logP, X=X, Z=Z)
            else:
                res[:, success] = self.tpt.evaluate(
                    logT=logT[success], logP=logP[success], X=X[success], Z=Z[success]
                )
                not_success = np.invert(success)

                res[i_logT, not_success] = logT[not_success]
                res[i_logRho, not_success] = logRho[not_success]
                res[i_logRho + 1 :, not_success] = np.nan
        return res
