import os
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import (
    RectBivariateSpline,
    RegularGridInterpolator,
    SmoothBivariateSpline,
)

from tinyeos.tableloader import TableLoader


class InterpolantsBuilder(TableLoader):
    """Builds the interpolants from the equation
    of state tables. The interpolants are bivariate
    spline approxmiations over a rectangular mesh
    as implemented with RectBivariatSpline. Interpolants are
    cached to the disks so they can be loaded later.

    Attributes:
        cache_path (str): interpolant cache path
        interpDT_var_x(y, z) (spline): (logT, logRho) interpolant of
            varialble var for hydrogen (x),
            helium (y) or the heavy element (z).
        interpPT_var_x(y, z) (spline): (logT, logP) interpolant
    """

    def __init__(self) -> None:
        """__init__ method. Initiates an instance of TableLoader
        and then builds all the interpolants.
        """

        # build DT interpolants with cms
        super().__init__(which_heavy="h2o", which_hhe="cms")
        self.__build_DT_x_interpolants()
        self.__build_DT_x_eff_interpolants()
        self.__build_DT_y_interpolants()
        self.__build_DT_z_interpolants()

        # build PT interpolants
        self.__build_PT_x_interpolants()
        self.__build_PT_x_eff_interpolants()
        self.__build_PT_y_interpolants()
        self.__build_PT_z_interpolants()

        # store interpolants to disk
        self.cache_path = Path(__file__).parent / "data/eos/interpolants"
        self.__cache_xy_interpolants("cms")
        self.__cache_z_interpolants("h2o")

        # cache DT interpolants for scvh
        super().__init__(which_heavy="h2o", which_hhe="scvh")
        self.__build_DT_x_interpolants()
        self.__build_DT_y_interpolants()

        # cache PT interpolants for scvh
        self.__build_PT_x_interpolants()
        self.__build_PT_y_interpolants()
        self.__cache_xy_interpolants("scvh")

        # cache interpolants for smoothed H2O
        super().__init__(which_heavy="h2o", use_smoothed_z_tables=True)
        self.__build_DT_z_interpolants()
        self.__build_PT_z_interpolants()
        self.__cache_z_interpolants("h2o_smoothed")

        # cache interpolants for SiO2, Fe, CO and the mixture
        for heavy_element in ["aqua", "sio2", "fe", "co", "mixture"]:
            super().__init__(which_heavy=heavy_element)
            self.__build_DT_z_interpolants()
            self.__build_PT_z_interpolants()
            self.__cache_z_interpolants(heavy_element)
            # smoothed version
            super().__init__(which_heavy=heavy_element, use_smoothed_z_tables=True)
            self.__build_DT_z_interpolants()
            self.__build_PT_z_interpolants()
            self.__cache_z_interpolants(heavy_element + "_smoothed")

    def __build_interpolant(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
        which_interpolant: bool = "rect",
        kx: int = 3,
        ky: int = 3,
    ):
        """Builds the interoplants with either RectBivariateSpline or
        SmoothBivariateSpline.

        Args:
            X, Y (ArrayLike): 1-D arrays of coordinates.
            Z (ArrayLike): 2-D array of data with shape (x.size, y.size)
            which_interpolant (str, optional): valid options are "rect"
                and "smooth". Defaults to "rect".
            kx, ky (ints, optional): degrees of the bivariate spline.

        Raises:
            NotImplementedError: raised if spline choice is not available.

        Returns:
            spline: fitted biviarate spline approximation.
        """

        # to-do: fix call for smooth interpolant
        if which_interpolant == "rect":
            Z = np.reshape(Z, (X.size, Y.size))
            spl = self.__build_rect_interpolant(X, Y, Z, kx, ky)
        elif which_interpolant == "smooth":
            spl = self.__build_smooth_interpolant(X, Y, Z, kx, ky)
        else:
            raise NotImplementedError("Spline choice not implemented.")
        return spl

    @staticmethod
    def __build_rect_interpolant(
        X: ArrayLike, Y: ArrayLike, Z: ArrayLike, kx: int = 3, ky: int = 3
    ) -> RectBivariateSpline:
        """Wrapper for RectBivariateSpline

        Args:
            X, Y (ArrayLike): 1-D arrays of coordinates.
            Z (ArrayLike): 2-D array of data with shape (x.size, y.size)
            kx, ky (ints, optional): degrees of the bivariate spline.
                Defaults to 3.

        Returns:
            RectBivariateSpline: fitted RectBivariateSpline.
        """
        return RectBivariateSpline(X, Y, Z, kx=kx, ky=ky)

    @staticmethod
    def __build_smooth_interpolant(
        X: ArrayLike, Y: ArrayLike, Z: ArrayLike, kx: int = 3, ky: int = 3
    ) -> SmoothBivariateSpline:
        """Wrapper for SmoothBivariateSpline

        Args:
            X, Y, Z (ArrayLike): 1-D sequences of data points
                (order is not important).
            kx, ky (ints, optional): degrees of the bivariate spline.
                defaults to 3.

        Returns:
            SmoothBivariateSpline: fitted SmoothBivariateSpline.
        """
        return SmoothBivariateSpline(X, Y, Z, kx=kx, ky=ky)

    @staticmethod
    def __build_grid_interpolant(
        points: Tuple, values: ArrayLike, method: str = "linear"
    ) -> RegularGridInterpolator:
        """Wrapper for RegularGridInterpolator

        Args:
            points (Tuple of ndarray of float, with shapes (m1,), ..., (mn,):
            The points defining the regular grid in n dimensions.
            values (ArrayLike, shape (m1, ..., mn, ...)):
            The data on the regular grid in n dimensions.
            method (str, optional): Interpolation method. Defaults to "linear".

        Returns:
            RegularGridInterpolator: fitted RegularGridInterpolor.
        """
        # to-do: reshape points and values
        # points = (logTs, logRhos)
        # shape(logTs) = (nlogT,)
        # shape(logRhos) = (nlogRho,)
        # shape(vals) = (nlogT, nlogRho)
        return RegularGridInterpolator(
            points, values, method=method, bounds_error=True, fill_value=np.nan
        )

    def __cache_interpolant(self, filename: str, obj: object) -> None:
        """Caches interpolant to the disk.

        Args:
            filename (str): name of the cache file.
            obj (spline): spline object.
        """

        filename = filename + ".npy"
        dst = os.path.join(self.cache_path, filename)
        np.save(dst, obj)

    def __cache_xy_interpolants(self, which_hhe: str = "cms") -> None:
        """Stores all interpolants for hydrogen and helium
        to the disk.

        Args:
            which_hhe (str, optional): which hydrogen-helium equation of state
                to use. Defaults to "cms".
        """

        filename = "interpDT_x_" + which_hhe
        interp_array = np.array(
            [
                self.interpDT_logP_x,
                self.interpDT_logS_x,
                self.interpDT_logU_x,
                self.interpDT_dlRho_dlT_P_x,
                self.interpDT_dlRho_dlP_T_x,
                self.interpDT_dlS_dlT_P_x,
                self.interpDT_dlS_dlP_T_x,
                self.interpDT_grad_ad_x,
                self.interpDT_lfe_x,
                self.interpDT_mu_x,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        if which_hhe == "cms":
            filename = "interpDT_x_eff_" + which_hhe
            interp_array = np.array(
                [
                    self.interpDT_logP_x_eff,
                    self.interpDT_logS_x_eff,
                    self.interpDT_logU_x_eff,
                    self.interpDT_dlRho_dlT_P_x_eff,
                    self.interpDT_dlRho_dlP_T_x_eff,
                    self.interpDT_dlS_dlT_P_x_eff,
                    self.interpDT_dlS_dlP_T_x_eff,
                    self.interpDT_grad_ad_x_eff,
                    self.interpDT_lfe_x_eff,
                    self.interpDT_mu_x_eff,
                ]
            )
            self.__cache_interpolant(filename, interp_array)

        filename = "interpPT_x_" + which_hhe
        interp_array = np.array(
            [
                self.interpPT_logRho_x,
                self.interpPT_logS_x,
                self.interpPT_logU_x,
                self.interpPT_dlRho_dlT_P_x,
                self.interpPT_dlRho_dlP_T_x,
                self.interpPT_dlS_dlT_P_x,
                self.interpPT_dlS_dlP_T_x,
                self.interpPT_grad_ad_x,
                self.interpPT_lfe_x,
                self.interpPT_mu_x,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        if which_hhe == "cms":
            filename = "interpPT_x_eff_" + which_hhe
            interp_array = np.array(
                [
                    self.interpPT_logRho_x_eff,
                    self.interpPT_logS_x_eff,
                    self.interpPT_logU_x_eff,
                    self.interpPT_dlRho_dlT_P_x_eff,
                    self.interpPT_dlRho_dlP_T_x_eff,
                    self.interpPT_dlS_dlT_P_x_eff,
                    self.interpPT_dlS_dlP_T_x_eff,
                    self.interpPT_grad_ad_x_eff,
                    self.interpPT_lfe_x_eff,
                    self.interpPT_mu_x_eff,
                ]
            )
            self.__cache_interpolant(filename, interp_array)

        filename = "interpDT_y_" + which_hhe
        interp_array = np.array(
            [
                self.interpDT_logP_y,
                self.interpDT_logS_y,
                self.interpDT_logU_y,
                self.interpDT_dlRho_dlT_P_y,
                self.interpDT_dlRho_dlP_T_y,
                self.interpDT_dlS_dlT_P_y,
                self.interpDT_dlS_dlP_T_y,
                self.interpDT_grad_ad_y,
                self.interpDT_lfe_y,
                self.interpDT_mu_y,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        filename = "interpPT_y_" + which_hhe
        interp_array = np.array(
            [
                self.interpPT_logRho_y,
                self.interpPT_logS_y,
                self.interpPT_logU_y,
                self.interpPT_dlRho_dlT_P_y,
                self.interpPT_dlRho_dlP_T_y,
                self.interpPT_dlS_dlT_P_y,
                self.interpPT_dlS_dlP_T_y,
                self.interpPT_grad_ad_y,
                self.interpPT_lfe_y,
                self.interpPT_mu_y,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

    def __cache_z_interpolants(self, which_heavy: str) -> None:
        """Stores all interpolants for the heavy element.
        to the disk.

        Args:
            which_heavy (str): which equation of state to use
                for the heavy element.
        """
        filename = "interpDT_z_" + which_heavy
        interp_array = np.array(
            [
                self.interpDT_logP_z,
                self.interpDT_logS_z,
                self.interpDT_logU_z,
                self.interpDT_grad_ad_z,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        filename = "interpPT_z_" + which_heavy
        interp_array = np.array(
            [
                self.interpPT_logRho_z,
                self.interpPT_logS_z,
                self.interpPT_logU_z,
                self.interpPT_grad_ad_z,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

    def __build_DT_x_interpolants(self) -> None:
        """Builds (logT, logRho) interpolants for hydrogen."""
        logT = self.x_DT_table[:, 0]
        logRho = self.x_DT_table[:, 2]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.x_DT_table[:, 1]
        self.interpDT_logP_x = self.__build_interpolant(X, Y, logP)

        logS = self.x_DT_table[:, 4]
        self.interpDT_logS_x = self.__build_interpolant(X, Y, logS)

        logU = self.x_DT_table[:, 3]
        self.interpDT_logU_x = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_DT_table[:, 5]
        self.interpDT_dlRho_dlT_P_x = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_DT_table[:, 6]
        self.interpDT_dlRho_dlP_T_x = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_DT_table[:, 7]
        self.interpDT_dlS_dlT_P_x = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_DT_table[:, 8]
        self.interpDT_dlS_dlP_T_x = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_DT_table[:, 9]
        self.interpDT_grad_ad_x = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_DT_table[:, 10]
        self.interpDT_lfe_x = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_DT_table[:, 11]
        self.interpDT_mu_x = self.__build_interpolant(X, Y, mu)

    def __build_DT_x_eff_interpolants(self) -> None:
        """Builds (logT, logRho) interpolants for "effective" hydrogen."""
        logT = self.x_eff_DT_table[:, 0]
        logRho = self.x_eff_DT_table[:, 2]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.x_eff_DT_table[:, 1]
        self.interpDT_logP_x_eff = self.__build_interpolant(X, Y, logP)

        logS = self.x_eff_DT_table[:, 4]
        self.interpDT_logS_x_eff = self.__build_interpolant(X, Y, logS)

        logU = self.x_eff_DT_table[:, 3]
        self.interpDT_logU_x_eff = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_eff_DT_table[:, 5]
        self.interpDT_dlRho_dlT_P_x_eff = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_eff_DT_table[:, 6]
        self.interpDT_dlRho_dlP_T_x_eff = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_eff_DT_table[:, 7]
        self.interpDT_dlS_dlT_P_x_eff = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_eff_DT_table[:, 8]
        self.interpDT_dlS_dlP_T_x_eff = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_eff_DT_table[:, 9]
        self.interpDT_grad_ad_x_eff = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_eff_DT_table[:, 10]
        self.interpDT_lfe_x_eff = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_eff_DT_table[:, 11]
        self.interpDT_mu_x_eff = self.__build_interpolant(X, Y, mu)

    def __build_PT_x_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for hydrogen."""
        logT = self.x_PT_table[:, 0]
        logP = self.x_PT_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.x_PT_table[:, 2]
        self.interpPT_logRho_x = self.__build_interpolant(X, Y, logRho)

        logS = self.x_PT_table[:, 4]
        self.interpPT_logS_x = self.__build_interpolant(X, Y, logS)

        logU = self.x_PT_table[:, 3]
        self.interpPT_logU_x = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_PT_table[:, 5]
        self.interpPT_dlRho_dlT_P_x = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_PT_table[:, 6]
        self.interpPT_dlRho_dlP_T_x = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_PT_table[:, 7]
        self.interpPT_dlS_dlT_P_x = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_PT_table[:, 8]
        self.interpPT_dlS_dlP_T_x = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_PT_table[:, 9]
        self.interpPT_grad_ad_x = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_PT_table[:, 10]
        self.interpPT_lfe_x = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_PT_table[:, 11]
        self.interpPT_mu_x = self.__build_interpolant(X, Y, mu)

    def __build_PT_x_eff_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for "effective" hydrogen."""
        logT = self.x_eff_PT_table[:, 0]
        logP = self.x_eff_PT_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.x_eff_PT_table[:, 2]
        self.interpPT_logRho_x_eff = self.__build_interpolant(X, Y, logRho)

        logS = self.x_eff_PT_table[:, 4]
        self.interpPT_logS_x_eff = self.__build_interpolant(X, Y, logS)

        logU = self.x_eff_PT_table[:, 3]
        self.interpPT_logU_x_eff = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_eff_PT_table[:, 5]
        self.interpPT_dlRho_dlT_P_x_eff = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_eff_PT_table[:, 6]
        self.interpPT_dlRho_dlP_T_x_eff = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_eff_PT_table[:, 7]
        self.interpPT_dlS_dlT_P_x_eff = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_eff_PT_table[:, 8]
        self.interpPT_dlS_dlP_T_x_eff = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_eff_PT_table[:, 9]
        self.interpPT_grad_ad_x_eff = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_eff_PT_table[:, 10]
        self.interpPT_lfe_x_eff = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_eff_PT_table[:, 11]
        self.interpPT_mu_x_eff = self.__build_interpolant(X, Y, mu)

    def __build_DT_y_interpolants(self) -> None:
        """Builds (logRho, logT) interpolants for helium."""
        logT = self.y_DT_table[:, 0]
        logRho = self.y_DT_table[:, 2]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.y_DT_table[:, 1]
        self.interpDT_logP_y = self.__build_interpolant(X, Y, logP)

        logS = self.y_DT_table[:, 4]
        self.interpDT_logS_y = self.__build_interpolant(X, Y, logS)

        logU = self.y_DT_table[:, 3]
        self.interpDT_logU_y = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.y_DT_table[:, 5]
        self.interpDT_dlRho_dlT_P_y = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.y_DT_table[:, 6]
        self.interpDT_dlRho_dlP_T_y = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.y_DT_table[:, 7]
        self.interpDT_dlS_dlT_P_y = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.y_DT_table[:, 8]
        self.interpDT_dlS_dlP_T_y = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.y_DT_table[:, 9]
        self.interpDT_grad_ad_y = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.y_DT_table[:, 10]
        self.interpDT_lfe_y = self.__build_interpolant(X, Y, log_free_e)

        mu = self.y_DT_table[:, 11]
        self.interpDT_mu_y = self.__build_interpolant(X, Y, mu)

    def __build_PT_y_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for helium."""
        logT = self.y_PT_table[:, 0]
        logP = self.y_PT_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.y_PT_table[:, 2]
        self.interpPT_logRho_y = self.__build_interpolant(X, Y, logRho)

        logS = self.y_PT_table[:, 4]
        self.interpPT_logS_y = self.__build_interpolant(X, Y, logS)

        logU = self.y_PT_table[:, 3]
        self.interpPT_logU_y = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.y_PT_table[:, 5]
        self.interpPT_dlRho_dlT_P_y = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.y_PT_table[:, 6]
        self.interpPT_dlRho_dlP_T_y = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.y_PT_table[:, 7]
        self.interpPT_dlS_dlT_P_y = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.y_PT_table[:, 8]
        self.interpPT_dlS_dlP_T_y = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.y_PT_table[:, 9]
        self.interpPT_grad_ad_y = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.y_PT_table[:, 10]
        self.interpPT_lfe_y = self.__build_interpolant(X, Y, log_free_e)

        mu = self.y_PT_table[:, 11]
        self.interpPT_mu_y = self.__build_interpolant(X, Y, mu)

    def __build_DT_z_interpolants(self) -> None:
        """Builds (logT, logRho) interpolants for the heavy element."""

        logT = self.z_DT_table[:, 0]
        logRho = self.z_DT_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.z_DT_table[:, 2]
        self.interpDT_logP_z = self.__build_interpolant(X, Y, logP)

        logS = self.z_DT_table[:, 4]
        self.interpDT_logS_z = self.__build_interpolant(X, Y, logS)

        logU = self.z_DT_table[:, 3]
        self.interpDT_logU_z = self.__build_interpolant(X, Y, logU)

        grad_ad = self.z_DT_table[:, 5]
        self.interpDT_grad_ad_z = self.__build_interpolant(X, Y, grad_ad)

    def __build_PT_z_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for for the heavy element."""
        logT = self.z_PT_table[:, 0]
        logP = self.z_PT_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.z_PT_table[:, 2]
        self.interpPT_logRho_z = self.__build_interpolant(X, Y, logRho)

        logS = self.z_PT_table[:, 4]
        self.interpPT_logS_z = self.__build_interpolant(X, Y, logS)

        logU = self.z_PT_table[:, 3]
        self.interpPT_logU_z = self.__build_interpolant(X, Y, logU)

        grad_ad = self.z_PT_table[:, 5]
        self.interpPT_grad_ad_z = self.__build_interpolant(X, Y, grad_ad)
