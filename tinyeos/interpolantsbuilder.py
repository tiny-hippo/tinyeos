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
        interp_dt_var_x(y, z) (spline): (logT, logRho) interpolant of
            varialble var for hydrogen (x),
            helium (y) or the heavy element (z).
        interp_pt_var_x(y, z) (spline): (logT, logP) interpolant
    """

    def __init__(self) -> None:
        """__init__ method. Initiates an instance of TableLoader
        and then builds all the interpolants.
        """

        # build dt interpolants with cms
        super().__init__(which_heavy="h2o", which_hhe="cms")
        self.__build_dt_x_interpolants()
        self.__build_dt_x_eff_interpolants()
        self.__build_dt_y_interpolants()
        self.__build_dt_z_interpolants()

        # build dt interpolants
        self.__build_pt_x_interpolants()
        self.__build_pt_x_eff_interpolants()
        self.__build_pt_y_interpolants()
        self.__build_pt_xy_interaction_interpolants()
        self.__build_pt_z_interpolants()

        # store interpolants to disk
        self.cache_path = Path(__file__).parent / "data/eos/interpolants"
        self.__cache_xy_interpolants("cms")
        self.__cache_z_interpolants("h2o")

        # cache dt interpolants for scvh
        super().__init__(which_heavy="h2o", which_hhe="scvh")
        self.__build_dt_x_interpolants()
        self.__build_dt_y_interpolants()

        # cache dt interpolants for scvh
        self.__build_pt_x_interpolants()
        self.__build_pt_y_interpolants()
        self.__cache_xy_interpolants("scvh")

        # cache dt interpolants for scvh_extended
        super().__init__(which_heavy="h2o", which_hhe="scvh_extended")
        self.__build_dt_x_interpolants()
        self.__build_dt_y_interpolants()

        # cache pt interpolants for scvh_extended
        self.__build_pt_x_interpolants()
        self.__build_pt_y_interpolants()
        self.__cache_xy_interpolants("scvh_extended")

        # cache interpolants for smoothed H2O
        super().__init__(which_heavy="h2o", use_smoothed_z_tables=True)
        self.__build_dt_z_interpolants()
        self.__build_pt_z_interpolants()
        self.__cache_z_interpolants("h2o_smoothed")

        # cache interpolants for SiO2, Fe, and CO
        for heavy_element in ["aqua", "sio2", "fe", "co"]:
            super().__init__(which_heavy=heavy_element)
            self.__build_dt_z_interpolants()
            self.__build_pt_z_interpolants()
            self.__cache_z_interpolants(heavy_element)
            # smoothed version
            super().__init__(which_heavy=heavy_element, use_smoothed_z_tables=True)
            self.__build_dt_z_interpolants()
            self.__build_pt_z_interpolants()
            self.__cache_z_interpolants(heavy_element + "_smoothed")

    def __build_z_mixture_interpolants(self, Z1: float, Z2: float, Z3: float) -> None:
        """Builds (logT, logRho) and (logT, logP) interpolants for
        heavy-element mixtures of H2O, SiO2, and Fe.

        Args:
            Z1 (float): mass-fraction of the first heavy element.
            Z2 (float): mass-fraction of the second heavy element.
            Z3 (float): mass-fraction of the third heavy element.

        """
        super().__init__(which_heavy="mixture", Z1=Z1, Z2=Z2, Z3=Z3)
        fname = (
            f"h2o_{100 * Z1:02.0f}"
            + f"_sio2_{100 * Z2:02.0f}"
            + f"_fe_{100 * Z3:02.0f}"
        )
        self.__build_dt_z_interpolants()
        self.__build_pt_z_interpolants()
        self.__cache_z_interpolants(fname)

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
        np.save(file=dst, arr=obj)

    def __cache_xy_interpolants(self, which_hhe: str = "cms") -> None:
        """Stores all interpolants for hydrogen and helium
        to the disk.

        Args:
            which_hhe (str, optional): which hydrogen-helium equation of state
                to use. Defaults to "cms".
        """

        filename = "interp_dt_x_" + which_hhe
        interp_array = np.array(
            [
                self.interp_dt_logP_x,
                self.interp_dt_logS_x,
                self.interp_dt_logU_x,
                self.interp_dt_dlRho_dlT_P_x,
                self.interp_dt_dlRho_dlP_T_x,
                self.interp_dt_dlS_dlT_P_x,
                self.interp_dt_dlS_dlP_T_x,
                self.interp_dt_grad_ad_x,
                self.interp_dt_lfe_x,
                self.interp_dt_mu_x,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        if which_hhe == "cms":
            filename = "interp_dt_x_eff_" + which_hhe
            interp_array = np.array(
                [
                    self.interp_dt_logP_x_eff,
                    self.interp_dt_logS_x_eff,
                    self.interp_dt_logU_x_eff,
                    self.interp_dt_dlRho_dlT_P_x_eff,
                    self.interp_dt_dlRho_dlP_T_x_eff,
                    self.interp_dt_dlS_dlT_P_x_eff,
                    self.interp_dt_dlS_dlP_T_x_eff,
                    self.interp_dt_grad_ad_x_eff,
                    self.interp_dt_lfe_x_eff,
                    self.interp_dt_mu_x_eff,
                ]
            )
            self.__cache_interpolant(filename, interp_array)

        filename = "interp_pt_x_" + which_hhe
        interp_array = np.array(
            [
                self.interp_pt_logRho_x,
                self.interp_pt_logS_x,
                self.interp_pt_logU_x,
                self.interp_pt_dlRho_dlT_P_x,
                self.interp_pt_dlRho_dlP_T_x,
                self.interp_pt_dlS_dlT_P_x,
                self.interp_pt_dlS_dlP_T_x,
                self.interp_pt_grad_ad_x,
                self.interp_pt_lfe_x,
                self.interp_pt_mu_x,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        if which_hhe == "cms":
            filename = "interp_pt_x_eff_" + which_hhe
            interp_array = np.array(
                [
                    self.interp_pt_logRho_x_eff,
                    self.interp_pt_logS_x_eff,
                    self.interp_pt_logU_x_eff,
                    self.interp_pt_dlRho_dlT_P_x_eff,
                    self.interp_pt_dlRho_dlP_T_x_eff,
                    self.interp_pt_dlS_dlT_P_x_eff,
                    self.interp_pt_dlS_dlP_T_x_eff,
                    self.interp_pt_grad_ad_x_eff,
                    self.interp_pt_lfe_x_eff,
                    self.interp_pt_mu_x_eff,
                ]
            )
            self.__cache_interpolant(filename, interp_array)

        filename = "interp_dt_y_" + which_hhe
        interp_array = np.array(
            [
                self.interp_dt_logP_y,
                self.interp_dt_logS_y,
                self.interp_dt_logU_y,
                self.interp_dt_dlRho_dlT_P_y,
                self.interp_dt_dlRho_dlP_T_y,
                self.interp_dt_dlS_dlT_P_y,
                self.interp_dt_dlS_dlP_T_y,
                self.interp_dt_grad_ad_y,
                self.interp_dt_lfe_y,
                self.interp_dt_mu_y,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        filename = "interp_pt_y_" + which_hhe
        interp_array = np.array(
            [
                self.interp_pt_logRho_y,
                self.interp_pt_logS_y,
                self.interp_pt_logU_y,
                self.interp_pt_dlRho_dlT_P_y,
                self.interp_pt_dlRho_dlP_T_y,
                self.interp_pt_dlS_dlT_P_y,
                self.interp_pt_dlS_dlP_T_y,
                self.interp_pt_grad_ad_y,
                self.interp_pt_lfe_y,
                self.interp_pt_mu_y,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        filename = "interp_pt_xy_int"
        interp_array = np.array([self.interp_pt_V_mix_xy, self.interp_pt_S_mix_xy])
        self.__cache_interpolant(filename, interp_array)

    def __cache_z_interpolants(self, which_heavy: str) -> None:
        """Stores all interpolants for the heavy element.
        to the disk.

        Args:
            which_heavy (str): which equation of state to use
                for the heavy element.
        """
        filename = "interp_dt_z_" + which_heavy
        interp_array = np.array(
            [
                self.interp_dt_logP_z,
                self.interp_dt_logS_z,
                self.interp_dt_logU_z,
                self.interp_dt_grad_ad_z,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

        filename = "interp_pt_z_" + which_heavy
        interp_array = np.array(
            [
                self.interp_pt_logRho_z,
                self.interp_pt_logS_z,
                self.interp_pt_logU_z,
                self.interp_pt_grad_ad_z,
            ]
        )
        self.__cache_interpolant(filename, interp_array)

    def __build_dt_x_interpolants(self) -> None:
        """Builds (logT, logRho) interpolants for hydrogen."""
        logT = self.x_dt_table[:, 0]
        logRho = self.x_dt_table[:, 2]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.x_dt_table[:, 1]
        self.interp_dt_logP_x = self.__build_interpolant(X, Y, logP)

        logS = self.x_dt_table[:, 4]
        self.interp_dt_logS_x = self.__build_interpolant(X, Y, logS)

        logU = self.x_dt_table[:, 3]
        self.interp_dt_logU_x = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_dt_table[:, 5]
        self.interp_dt_dlRho_dlT_P_x = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_dt_table[:, 6]
        self.interp_dt_dlRho_dlP_T_x = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_dt_table[:, 7]
        self.interp_dt_dlS_dlT_P_x = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_dt_table[:, 8]
        self.interp_dt_dlS_dlP_T_x = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_dt_table[:, 9]
        self.interp_dt_grad_ad_x = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_dt_table[:, 10]
        self.interp_dt_lfe_x = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_dt_table[:, 11]
        self.interp_dt_mu_x = self.__build_interpolant(X, Y, mu)

    def __build_dt_x_eff_interpolants(self) -> None:
        """Builds (logT, logRho) interpolants for "effective" hydrogen."""
        logT = self.x_eff_dt_table[:, 0]
        logRho = self.x_eff_dt_table[:, 2]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.x_eff_dt_table[:, 1]
        self.interp_dt_logP_x_eff = self.__build_interpolant(X, Y, logP)

        logS = self.x_eff_dt_table[:, 4]
        self.interp_dt_logS_x_eff = self.__build_interpolant(X, Y, logS)

        logU = self.x_eff_dt_table[:, 3]
        self.interp_dt_logU_x_eff = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_eff_dt_table[:, 5]
        self.interp_dt_dlRho_dlT_P_x_eff = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_eff_dt_table[:, 6]
        self.interp_dt_dlRho_dlP_T_x_eff = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_eff_dt_table[:, 7]
        self.interp_dt_dlS_dlT_P_x_eff = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_eff_dt_table[:, 8]
        self.interp_dt_dlS_dlP_T_x_eff = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_eff_dt_table[:, 9]
        self.interp_dt_grad_ad_x_eff = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_eff_dt_table[:, 10]
        self.interp_dt_lfe_x_eff = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_eff_dt_table[:, 11]
        self.interp_dt_mu_x_eff = self.__build_interpolant(X, Y, mu)

    def __build_pt_x_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for hydrogen."""
        logT = self.x_pt_table[:, 0]
        logP = self.x_pt_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.x_pt_table[:, 2]
        self.interp_pt_logRho_x = self.__build_interpolant(X, Y, logRho)

        logS = self.x_pt_table[:, 4]
        self.interp_pt_logS_x = self.__build_interpolant(X, Y, logS)

        logU = self.x_pt_table[:, 3]
        self.interp_pt_logU_x = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_pt_table[:, 5]
        self.interp_pt_dlRho_dlT_P_x = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_pt_table[:, 6]
        self.interp_pt_dlRho_dlP_T_x = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_pt_table[:, 7]
        self.interp_pt_dlS_dlT_P_x = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_pt_table[:, 8]
        self.interp_pt_dlS_dlP_T_x = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_pt_table[:, 9]
        self.interp_pt_grad_ad_x = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_pt_table[:, 10]
        self.interp_pt_lfe_x = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_pt_table[:, 11]
        self.interp_pt_mu_x = self.__build_interpolant(X, Y, mu)

    def __build_pt_x_eff_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for "effective" hydrogen."""
        logT = self.x_eff_pt_table[:, 0]
        logP = self.x_eff_pt_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.x_eff_pt_table[:, 2]
        self.interp_pt_logRho_x_eff = self.__build_interpolant(X, Y, logRho)

        logS = self.x_eff_pt_table[:, 4]
        self.interp_pt_logS_x_eff = self.__build_interpolant(X, Y, logS)

        logU = self.x_eff_pt_table[:, 3]
        self.interp_pt_logU_x_eff = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.x_eff_pt_table[:, 5]
        self.interp_pt_dlRho_dlT_P_x_eff = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.x_eff_pt_table[:, 6]
        self.interp_pt_dlRho_dlP_T_x_eff = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.x_eff_pt_table[:, 7]
        self.interp_pt_dlS_dlT_P_x_eff = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.x_eff_pt_table[:, 8]
        self.interp_pt_dlS_dlP_T_x_eff = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.x_eff_pt_table[:, 9]
        self.interp_pt_grad_ad_x_eff = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.x_eff_pt_table[:, 10]
        self.interp_pt_lfe_x_eff = self.__build_interpolant(X, Y, log_free_e)

        mu = self.x_eff_pt_table[:, 11]
        self.interp_pt_mu_x_eff = self.__build_interpolant(X, Y, mu)

    def __build_dt_y_interpolants(self) -> None:
        """Builds (logRho, logT) interpolants for helium."""
        logT = self.y_dt_table[:, 0]
        logRho = self.y_dt_table[:, 2]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.y_dt_table[:, 1]
        self.interp_dt_logP_y = self.__build_interpolant(X, Y, logP)

        logS = self.y_dt_table[:, 4]
        self.interp_dt_logS_y = self.__build_interpolant(X, Y, logS)

        logU = self.y_dt_table[:, 3]
        self.interp_dt_logU_y = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.y_dt_table[:, 5]
        self.interp_dt_dlRho_dlT_P_y = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.y_dt_table[:, 6]
        self.interp_dt_dlRho_dlP_T_y = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.y_dt_table[:, 7]
        self.interp_dt_dlS_dlT_P_y = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.y_dt_table[:, 8]
        self.interp_dt_dlS_dlP_T_y = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.y_dt_table[:, 9]
        self.interp_dt_grad_ad_y = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.y_dt_table[:, 10]
        self.interp_dt_lfe_y = self.__build_interpolant(X, Y, log_free_e)

        mu = self.y_dt_table[:, 11]
        self.interp_dt_mu_y = self.__build_interpolant(X, Y, mu)

    def __build_pt_y_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for helium."""
        logT = self.y_pt_table[:, 0]
        logP = self.y_pt_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.y_pt_table[:, 2]
        self.interp_pt_logRho_y = self.__build_interpolant(X, Y, logRho)

        logS = self.y_pt_table[:, 4]
        self.interp_pt_logS_y = self.__build_interpolant(X, Y, logS)

        logU = self.y_pt_table[:, 3]
        self.interp_pt_logU_y = self.__build_interpolant(X, Y, logU)

        dlRho_dlT_P = self.y_pt_table[:, 5]
        self.interp_pt_dlRho_dlT_P_y = self.__build_interpolant(X, Y, dlRho_dlT_P)

        dlRho_dlP_T = self.y_pt_table[:, 6]
        self.interp_pt_dlRho_dlP_T_y = self.__build_interpolant(X, Y, dlRho_dlP_T)

        dlS_dlT_P = self.y_pt_table[:, 7]
        self.interp_pt_dlS_dlT_P_y = self.__build_interpolant(X, Y, dlS_dlT_P)

        dlS_dlP_T = self.y_pt_table[:, 8]
        self.interp_pt_dlS_dlP_T_y = self.__build_interpolant(X, Y, dlS_dlP_T)

        grad_ad = self.y_pt_table[:, 9]
        self.interp_pt_grad_ad_y = self.__build_interpolant(X, Y, grad_ad)

        log_free_e = self.y_pt_table[:, 10]
        self.interp_pt_lfe_y = self.__build_interpolant(X, Y, log_free_e)

        mu = self.y_pt_table[:, 11]
        self.interp_pt_mu_y = self.__build_interpolant(X, Y, mu)

    def __build_pt_xy_interaction_interpolants(self) -> None:
        """Builds (logP, logT) interpolants for the
        hydrogen-helium non-ideal interaction.
        """
        logT = self.xy_interaction_pt_table[:, 1]
        logP = self.xy_interaction_pt_table[:, 0]
        V_mix = self.xy_interaction_pt_table[:, 2]
        S_mix = self.xy_interaction_pt_table[:, 3]

        X = np.unique(logP)
        Y = np.unique(logT)
        self.interp_pt_V_mix_xy = self.__build_interpolant(X, Y, V_mix)
        self.interp_pt_S_mix_xy = self.__build_interpolant(X, Y, S_mix)
        # points = (X, Y)
        # values = np.reshape(V_mix, (X.size, Y.size))
        # self.interp_pt_V_mix_xy = RegularGridInterpolator(points, values, method="linear")
        # values = np.reshape(S_mix, (X.size, Y.size))
        # self.interp_pt_S_mix_xy = RegularGridInterpolator(points, values, method="linear")

    def __build_dt_z_interpolants(self) -> None:
        """Builds (logT, logRho) interpolants for the heavy element."""

        logT = self.z_dt_table[:, 0]
        logRho = self.z_dt_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logRho)

        logP = self.z_dt_table[:, 2]
        self.interp_dt_logP_z = self.__build_interpolant(X, Y, logP)

        logS = self.z_dt_table[:, 4]
        self.interp_dt_logS_z = self.__build_interpolant(X, Y, logS)

        logU = self.z_dt_table[:, 3]
        self.interp_dt_logU_z = self.__build_interpolant(X, Y, logU)

        grad_ad = self.z_dt_table[:, 5]
        self.interp_dt_grad_ad_z = self.__build_interpolant(X, Y, grad_ad)

    def __build_pt_z_interpolants(self) -> None:
        """Builds (logT, logP) interpolants for for the heavy element."""
        logT = self.z_pt_table[:, 0]
        logP = self.z_pt_table[:, 1]
        X = np.unique(logT)
        Y = np.unique(logP)

        logRho = self.z_pt_table[:, 2]
        self.interp_pt_logRho_z = self.__build_interpolant(X, Y, logRho)

        logS = self.z_pt_table[:, 4]
        self.interp_pt_logS_z = self.__build_interpolant(X, Y, logS)

        logU = self.z_pt_table[:, 3]
        self.interp_pt_logU_z = self.__build_interpolant(X, Y, logU)

        grad_ad = self.z_pt_table[:, 5]
        self.interp_pt_grad_ad_z = self.__build_interpolant(X, Y, grad_ad)
