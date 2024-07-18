import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import NearestNDInterpolator, PchipInterpolator, interp1d


class TableLoader:
    """Loads the equation of state tables for hydrogen, helium
    and a heavy element.g

    Equations of state implemented:
        Hydrogen-Helium:
            CMS (Chabrier et al. 2019),
            SCvH (Saumon et al. 1995),
            SCvH extended version (R. Helled, priv. comm.).

        Heavy element:
            H2O (QEOS, More et al. 1988 and AQUA, Haldemann et al. 2020),
            SiO2 (QEOS, More et al. 1988),
            Fe (QEOS, More et al. 1998),
            CO (QEOS, Podolak et al. 2022),
            ideal mixture of H2O and SiO2 (QEOS, More et al. 1988).

    Attributes:
        x_DT_table (ndarray): hydrogen (logRho, logT) table
        x_PT_table (ndarray): hydrogen (logP, logT) table
        x_eff_PT_table (ndarray): (logP, logT) table
        y_DT_table (ndarray): helium (logRho, logT) table
        y_PT_table (ndarray): helium (logP, logT) table
        z_DT_table (ndarray): heavy-element (logRho, logT) table
        z_PT_table (ndarray): heavy-element (logP, logT) table
    """

    def __init__(
        self,
        which_hhe: str = "cms",
        which_heavy: str = "h2o",
        Z1: float = 0.5,
        Z2: float = 0.5,
        Z3: float = 0,
        use_smoothed_xy_tables: bool = False,
        use_smoothed_z_tables: bool = False,
    ) -> None:
        """__init__ method. Sets the equation of state
        boundaries and loads the tables.

        Args:
            which_hhe (str, optional): which hydrogen-helium equation of state
                to use. Defaults to "cms".
            which_heavy (str, optional): which heavy-element equation of state
                to use. Defaults to "h2o".
            Z1 (float, optional): mass-fraction of the first heavy element.
                Defaults to 0.5
            Z2 (float, optional): mass-fraction of the second heavy element.
                Defaults to 0.5.
            Z3 (float, optional): mass-fraction of the third heavy element.
                Defaults to 0.
            use_smoothed_xy_tables (bool, optional): whether to use smoothed
                hydrogen and helium tables. Defaults to False.
            use_smoothed_z_tables (bool, optional): whether to use smoothed
                heavy-element tables. Defaults to False.
        """
        self.use_smoothed_xy_tables = use_smoothed_xy_tables
        self.use_smoothed_z_tables = use_smoothed_z_tables
        self.tables_path = Path(__file__).parent / "data/eos/tables"
        self.z_dt_header = (
            "logT [K] logRho [g/cc] logP [Ba] logU [erg/g] logS [erg/g/K] grad_ad"
        )
        self.z_pt_header = (
            "logT [K] logP [Ba] logRho [g/cc] logU [erg/g] logS [erg/g/K] grad_ad"
        )

        if which_heavy == "mixture" and not np.isclose(Z1 + Z2 + Z3, 1, atol=1e-5):
            msg = "Mass fractions of the heavy elements must sum to one."
            raise ValueError(msg)

        _, _ = self.__load_xy_dt_tables(which_hhe=which_hhe)
        _, _ = self.__load_xy_pt_tables(which_hhe=which_hhe)
        _ = self.__load_xy_pt_interaction_tables()
        _ = self.__load_z_dt_table(which_heavy=which_heavy, Z1=Z1, Z2=Z2, Z3=Z3)
        _ = self.__load_z_pt_table(which_heavy=which_heavy, Z1=Z1, Z2=Z2, Z3=Z3)

    def __load_xy_dt_tables(self, which_hhe: str = "cms") -> Tuple[NDArray, NDArray]:
        """Loads the hydrogen and helium (logRho, logT) tables.

        Args:
            which_hhe (str, optional): which hydrogen-helium equation of state
                to use. Defaults to "cms".

        Raises:
            NotImplementedError: raised if which_hhe option is unavailable.

        Returns:
            Tuple[NDArray, NDArray]: hydrogen and helium tables
        """

        if which_hhe == "cms":
            src = os.path.join(self.tables_path, "cms_dt_hydrogen.pkl")
        elif which_hhe == "scvh" or which_hhe == "scvh_extended":
            src = os.path.join(self.tables_path, "scvh_extended_dt_hydrogen.pkl")
        else:
            raise NotImplementedError("this table is not available.")
        with open(src, "rb") as file:
            data = pickle.load(file)
        # columns = ["logT", "logP", "logRho", "logU", "logS", "dlnRho/dlnT",
        #          "dlnRho/dlnP", "dlnS/dlnT", "dlnS/dlnP", "grad_ad",
        #          "log_free_e", "mu"]
        self.x_DT_table = data

        # load the "effective" hydrogen table from Chabrier et al. 2021
        # that can be used to account for hydrogen-helium interactions
        if which_hhe == "cms":
            src = os.path.join(self.tables_path, "cms_dt_effective_hydrogen.pkl")
            with open(src, "rb") as file:
                data = pickle.load(file)
            self.x_eff_DT_table = data

        if which_hhe == "cms":
            src = os.path.join(self.tables_path, "cms_dt_helium.pkl")
        elif which_hhe == "scvh" or which_hhe == "scvh_extended":
            src = os.path.join(self.tables_path, "scvh_extended_dt_helium.pkl")
        with open(src, "rb") as file:
            data = pickle.load(file)
        self.y_DT_table = data
        return (self.x_DT_table, self.y_DT_table)

    def __load_xy_pt_tables(self, which_hhe: str = "cms") -> Tuple[NDArray, NDArray]:
        """Loads the hydrogen and helium (logP, logT) tables.

        Args:
            which_hhe (str, optional):  Which hydrogen-helium equation of state
                to use. Defaults to "cms".

        Raises:
            NotImplementedError: raised if which_hhe option is unvailable.

        Returns:
            Tuple[NDArray, NDArray]: hydrogen and helium tables
        """

        if which_hhe == "cms":
            src = os.path.join(self.tables_path, "cms_pt_hydrogen.pkl")
        elif which_hhe == "scvh":
            src = os.path.join(self.tables_path, "scvh_pt_hydrogen.pkl")
        elif which_hhe == "scvh_extended":
            src = os.path.join(self.tables_path, "scvh_extended_pt_hydrogen.pkl")
        else:
            raise NotImplementedError("this table is not available.")
        with open(src, "rb") as file:
            data = pickle.load(file)
        # columns = ["logT", "logP", "logRho", "logU", "logS", "dlnRho/dlnT",
        #          "dlnRho/dlnP", "dlnS/dlnT", "dlnS/dlnP", "grad_ad",
        #          "log_free_e", "mu"]
        self.x_PT_table = data

        # load the "effective" hydrogen table from Chabrier et al. 2021
        # that can be used to account for hydrogen-helium interactions
        if which_hhe == "cms":
            src = os.path.join(self.tables_path, "cms_pt_effective_hydrogen.pkl")
            with open(src, "rb") as file:
                data = pickle.load(file)
            self.x_eff_PT_table = data

        if which_hhe == "cms":
            src = os.path.join(self.tables_path, "cms_pt_helium.pkl")
        elif which_hhe == "scvh":
            src = os.path.join(self.tables_path, "scvh_pt_helium.pkl")
        elif which_hhe == "scvh_extended":
            src = os.path.join(self.tables_path, "scvh_extended_pt_helium.pkl")
        with open(src, "rb") as file:
            data = pickle.load(file)
        self.y_PT_table = data
        return (self.x_PT_table, self.y_PT_table)

    def __load_xy_pt_interaction_tables(self) -> NDArray:
        """Loads the hydrogen-helium non-ideal interaction table
        from Howard & Guillot (2023).

        Returns:
            NDArray: hydrogen-helium non-ideal interaction table
        """
        fname = "pt_hydrogen_helium_interactions.data"
        src = os.path.join(self.tables_path, fname)
        # columns = ["logP", "logT", "V_mix", "S_mix"]
        data = np.loadtxt(src)
        self.xy_interaction_PT_table = data
        return self.xy_interaction_PT_table

    def __load_z_dt_table(
        self, which_heavy: str, Z1: float = 0.5, Z2: float = 0.5, Z3: float = 0
    ) -> NDArray:
        """Loads the heavy-element (logRho, logT) tables.
        For the qeos heavy-element tables, the adiabatic gradient
        column is filled with dummy values that are not used.

        Args:
            which_heavy (str): which heavy-element equation of state to use.
            Z1 (float, optional): mass-fraction of the first heavy element.
                Defaults to 0.5
            Z2 (float, optional): mass-fraction of the second heavy element.
                Defaults to 0.5.
            Z3 (float, optional): mass-fraction of the third heavy element.
                Defaults to 0.

        Raises:
            NotImplementedError: raised if which_heavy option is unavailable.

        Returns:
            NDArray: the heavy-element table
        """
        extra = "smoothed_" if self.use_smoothed_z_tables else ""
        if which_heavy == "h2o":
            fname = f"qeos_{extra}dt_h2o.data"
        elif which_heavy == "aqua":
            fname = "aqua_dt_h2o.data"
        elif which_heavy == "sio2":
            fname = f"qeos_{extra}dt_sio2.data"
        elif which_heavy == "mixture":
            fname = (
                f"qeos_{extra}dt_h2o_{100 * Z1:02.0f}"
                + f"_sio2_{100 * Z2:02.0f}"
                + f"_fe_{100 * Z3:02.0f}.data"
            )
            if not os.path.exists(os.path.join(self.tables_path, fname)):
                self.mix_heavy_elements(
                    which_Z1="h2o",
                    Z1=Z1,
                    which_Z2="sio2",
                    Z2=Z2,
                    which_Z3="fe",
                    Z3=Z3,
                    store_table=True,
                )
                self.invert_z_table(
                    which_variables="dt",
                    which_heavy="mixture",
                    Z1=Z1,
                    Z2=Z2,
                    Z3=Z3,
                    kind="pchip",
                    extrapolate=True,
                    smooth_table=True,
                    num_smoothing_rounds=1,
                    store_table=True,
                )
        elif which_heavy == "fe":
            fname = f"qeos_{extra}dt_fe.data"
        elif which_heavy == "co":
            fname = f"qeos_{extra}dt_co.data"
        else:
            raise NotImplementedError("this heavy element is not available")

        src = os.path.join(self.tables_path, fname)
        data = np.loadtxt(src, skiprows=1, dtype=np.float64)
        data = np.loadtxt(src, skiprows=1, dtype=np.float64)
        # columns = ["logT", "logRho", "logP", "logU", "logS", "grad_ad"]
        self.z_DT_table = data
        return self.z_DT_table

    def __load_z_pt_table(
        self, which_heavy: str, Z1: float = 0.5, Z2: float = 0.5, Z3: float = 0
    ) -> NDArray:
        """Loads the heavy-element (logP, logT) tables.
        For the qeos heavy-element tables, the adiabatic gradient
        column is filled with dummy values that are not used.

        Args:
            which_heavy (str): which heavy-element equation of state to use.
            Z1 (float, optional): mass-fraction of the first heavy element.
                Defaults to 0.5
            Z2 (float, optional): mass-fraction of the second heavy element.
                Defaults to 0.5.
            Z3 (float, optional): mass-fraction of the third heavy element.
                Defaults to 0.

        Raises:
            NotImplementedError: raised if which_heavy option is unavailable.

        Returns:
            NDArray: the heavy-element table
        """
        extra = "smoothed_" if self.use_smoothed_z_tables else ""
        if which_heavy == "h2o":
            fname = f"qeos_{extra}pt_h2o.data"
        elif which_heavy == "aqua":
            fname = "aqua_pt_h2o.data"
        elif which_heavy == "sio2":
            fname = f"qeos_{extra}pt_sio2.data"
        elif which_heavy == "mixture":
            fname = (
                f"qeos_{extra}pt_h2o_{100 * Z1:02.0f}"
                + f"_sio2_{100 * Z2:02.0f}"
                + f"_fe_{100 * Z3:02.0f}.data"
            )
            if not os.path.exists(os.path.join(self.tables_path, fname)):
                self.mix_heavy_elements(
                    which_Z1="h2o",
                    Z1=Z1,
                    which_Z2="sio2",
                    Z2=Z2,
                    which_Z3="fe",
                    Z3=Z3,
                    store_table=True,
                )
        elif which_heavy == "fe":
            fname = f"qeos_{extra}pt_fe.data"
        elif which_heavy == "co":
            fname = f"qeos_{extra}pt_co.data"
        else:
            raise NotImplementedError("this heavy element is not available")
        src = os.path.join(self.tables_path, fname)
        data = np.loadtxt(src, skiprows=1, dtype=np.float64)
        # columns = ["logT", "logP", "logRho", "logU", "logS", "grad_ad"]
        self.z_PT_table = data
        return self.z_PT_table

    def invert_z_table(
        self,
        which_variables: str,
        which_heavy: str,
        Z1: float = 0.5,
        Z2: float = 0.5,
        Z3: float = 0,
        kind: str = "pchip",
        extrapolate: bool = True,
        smooth_table: bool = False,
        num_smoothing_rounds: int = 1,
        store_table: bool = False,
    ) -> NDArray:
        """Converts a qeos heavy-element (logT, logRho) table to
        (logT, logP) and optionally smoothes the results.
        Assumes that the qeos table is organised along isotherms.

        Args:
            which_variables (str): to which variables to convert.
            which_heavy (str): name of the heavy element.
                Current options are "h2o", "sio2", "fe" and "co".
            Z1 (float, optional): mass-fraction of the first heavy element.
                Defaults to 0.5
            Z2 (float, optional): mass-fraction of the second heavy element.
                Defaults to 0.5.
            Z3 (float, optional): mass-fraction of the third heavy element.
                Defaults to 0.
            kind (str, optional): interpolation method to use. Options
                are linear and cubic, and pchip. Defaults to pchip.
            extrapolate (bool, optional): whether to extrapolate for missing
                data. If False, uses a two-dimensional nearest
                neigh-neighbour extrapolation to replace
                the missing data. Defaults to True.
            smooth_table (bool, optional): whether to smooth the new table.
                Defaults to False.
            num_smoothing_rounds (int, optional): number of times to smooth
                the new table. Defaults to 1.
            store_table (bool, optional): whether to store the table.
                Defaults to False.

        Returns:
            NDArray: inverted heavy-element table
        """
        if which_variables == "pt":
            # indices: 0: logT, 1: logRho, 2: logP, 3: logU, 4: logS
            z_in_table = self.__load_z_dt_table(
                which_heavy=which_heavy, Z1=Z1, Z2=Z2, Z3=Z3
            )
            dlogP = 0.05
            y = np.arange(-3.8, 15.8 + dlogP, dlogP)
        elif which_variables == "dt":
            # indices: 0: logT, 1: logP, 2: logRho, 3: logU, 4: logS
            z_in_table = self.__load_z_pt_table(
                which_heavy=which_heavy, Z1=Z1, Z2=Z2, Z3=Z3
            )
            dlogRho = 0.05
            y = np.arange(-12, 2 + dlogRho, dlogRho)
        else:
            raise ValueError("Invalid which_variables option.")

        # (logT, var2) grid
        logT = z_in_table[:, 0]
        var2 = z_in_table[:, 2]  # logRho or logP
        x = np.unique(logT)
        num_xs = x.size
        num_ys = y.size
        num_vals = z_in_table.shape[1]

        # indices for the dependent variables of the new table
        which_values = [1, 3, 4, 5]
        # get the upper and lower bounds
        min_vals = np.min(z_in_table, axis=0)
        max_vals = np.max(z_in_table, axis=0)

        z_out_table = np.zeros((num_xs, num_ys, num_vals))
        for i, logT_isotherm in enumerate(x):
            # look for the points on the current isotherm
            k = logT == logT_isotherm
            var2_isotherm = var2[k]
            vals = z_in_table[k, :]
            # look for unique pressure or density points and select those values
            var2_isotherm, n = np.unique(var2_isotherm, return_index=True)
            vals = vals[n, :]
            # get rid of the independent variables
            vals = vals[:, which_values]

            # interpolate on the unique pressure or densities of the isotherm
            if kind == "linear" or kind == "cubic":
                fill_value = "extrapolate" if extrapolate else np.nan
                f = interp1d(
                    x=var2_isotherm,
                    y=np.transpose(vals),
                    kind=kind,
                    fill_value=fill_value,
                    bounds_error=False,
                )
                res = f(y)
            elif kind == "pchip":
                # monotonic cubic interpolation
                res = np.zeros((vals.shape[1], y.shape[0]))
                for k in range(vals.shape[1]):
                    which_val = vals[:, k]
                    f = PchipInterpolator(
                        x=var2_isotherm,
                        y=which_val,
                        extrapolate=extrapolate,
                    )
                    res[k, :] = f(y)
            else:
                raise ValueError("invalid interpolation method.")

            sub_table = np.zeros((z_out_table.shape[1], z_out_table.shape[2]))
            sub_table[:, 0] = logT_isotherm * np.ones_like(y)
            sub_table[:, 1] = y
            sub_table[:, 2:] = np.transpose(res)
            # enforce upper and lower bounds
            for n, k in enumerate(which_values):
                check_min = sub_table[:, n + 2] < min_vals[k]
                check_max = sub_table[:, n + 2] > max_vals[k]
                sub_table[check_min, n + 2] = min_vals[k]
                sub_table[check_max, n + 2] = max_vals[k]
            z_out_table[i] = sub_table

        if smooth_table:
            z_out_table = self.smooth_z_table(z_out_table, num_smoothing_rounds)
        out_table = z_out_table.reshape((-1, num_vals))

        # fill missing values with a 2d nearest neighbour extrapolation
        if not extrapolate:
            x = out_table[:, 0]  # logT
            y = out_table[:, 1]  # logP
            for i in range(2, out_table.shape[1]):
                z = out_table[:, i]
                mask = ~np.isnan(z)
                f = NearestNDInterpolator(list(zip(x[mask], y[mask])), z[mask])
                res = f(x, y)
                mask = np.isnan(z)
                z[mask] = res[mask]
                out_table[:, i] = z

        if which_heavy == "mixture":
            fname = (
                f"qeos_{which_variables}_h2o_{100 * Z1:02.0f}"
                + f"_sio2_{100 * Z2:02.0f}"
                + f"_fe_{100 * Z3:02.0f}.data"
            )
        else:
            fname = f"qeos_{which_variables}_{which_heavy}.data"
        if store_table:
            header = self.z_dt_header if which_variables == "dt" else self.z_pt_header
            dst = os.path.join(self.tables_path, fname)
            np.savetxt(dst, out_table, fmt="%.8e", header=header)
        return out_table

    def invert_xeff_pt_table(
        self,
        kind: str = "linear",
        extrapolate: bool = True,
        smooth_table: bool = False,
        num_smoothing_rounds: int = 1,
        store_table: bool = False,
    ) -> NDArray:
        """Converts the CMS effective hydrogen (logT, logP) table to
        (logT, logRho) and optionally smoothes the results.
        Assumes that the table is organised along isotherms.

        Args:
            kind (str): interpolation method to use. Options
                are linear and cubic. Defaults to linear.
            extrapolate (bool): whether to extrapolate for missing
                data. If False, uses a two-dimensional nearest
                neigh-neighbour extrapolation to replace
                the missing data. Defaults to True.
            smooth_table (bool, optional): whether to smooth the new table.
                Defaults to False.
            num_smoothing_rounds (int, optional): number of times to smooth
                the new table. Defaults to 1.
            store_table (bool, optional): whether to store the table.
                Defaults to False.

        Returns:
            NDArray: inverted effective hydrogen table
        """
        x_eff_PT_table = self.x_eff_PT_table
        logT = x_eff_PT_table[:, 0]
        logRho = x_eff_PT_table[:, 2]

        # (logT, logRho) grid
        x = np.unique(logT)
        dlogRho = 0.05
        # boundaries are close to min/max logRho
        # in the pressure-temperature tables
        y = np.arange(-9, 6 + dlogRho, dlogRho)
        num_xs = x.size
        num_ys = y.size
        num_vals = x_eff_PT_table.shape[1]

        # indices for the dependent variables of the new table
        which_values = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # get the upper and lower bounds
        min_vals = np.min(x_eff_PT_table, axis=0)
        max_vals = np.max(x_eff_PT_table, axis=0)

        x_eff_DT_table = np.zeros((num_xs, num_ys, num_vals))
        for i, logT_isotherm in enumerate(x):
            # look for the points on the current isotherm
            k = logT == logT_isotherm
            logRho_isotherm = logRho[k]
            vals = x_eff_PT_table[k, :]
            # look for unique logRho points and select those values
            logRho_isotherm, n = np.unique(logRho_isotherm, return_index=True)
            vals = vals[n, :]
            # get rid of logT and logRho
            vals = vals[:, which_values]

            # interpolate on the unique logRhos of the isotherm
            fill_value = "extrapolate" if extrapolate else np.nan
            f = interp1d(
                logRho_isotherm,
                np.transpose(vals),
                kind=kind,
                fill_value=fill_value,
                bounds_error=False,
            )
            res = f(y)

            sub_table = np.zeros((x_eff_DT_table.shape[1], x_eff_DT_table.shape[2]))
            sub_table[:, 0] = logT_isotherm * np.ones_like(y)
            sub_table[:, 1] = y
            sub_table[:, 2:] = np.transpose(res)
            # enforce upper and lower bounds
            for n, k in enumerate(which_values):
                check_min = sub_table[:, n + 2] < min_vals[k]
                check_max = sub_table[:, n + 2] > max_vals[k]
                sub_table[check_min, n + 2] = min_vals[k]
                sub_table[check_max, n + 2] = max_vals[k]
            x_eff_DT_table[i] = sub_table

        fname = "cms_dt_effective_hydrogen.pkl"
        if smooth_table:
            fname = fname.replace("dt", "dt_smoothed")
            x_eff_DT_table = self.smooth_xy_table(
                x_eff_DT_table, "logRho", num_smoothing_rounds=num_smoothing_rounds
            )
        out_table = x_eff_DT_table.reshape((-1, num_vals))

        # fill missing values with a 2d nearest neighbour extrapolation
        if not extrapolate:
            x = out_table[:, 0]  # logT
            y = out_table[:, 1]  # logRho
            for i in range(2, out_table.shape[1]):
                z = out_table[:, i]
                mask = ~np.isnan(z)
                f = NearestNDInterpolator(list(zip(x[mask], y[mask])), z[mask])
                res = f(x, y)
                mask = np.isnan(z)
                z[mask] = res[mask]
                out_table[:, i] = z

        # CMS tables have the same order of the data
        # for pressure-temperature and density-temperature tables
        tmp_table = np.copy(out_table)
        out_table[:, 1] = tmp_table[:, 2]  # set logP
        out_table[:, 2] = tmp_table[:, 1]  # set logRho
        if store_table:
            dst = os.path.join(self.tables_path, fname)
            with open(dst, "wb") as file:
                pickle.dump(out_table, file)
        return out_table

    def mix_heavy_elements(
        self,
        which_Z1: str,
        Z1: float,
        which_Z2: str,
        Z2: float,
        which_Z3: str,
        Z3: float,
        store_table: bool = False,
    ) -> NDArray:
        """Uses the ideal mixing approximation to
        calculates the density, energy and entropy for a mixture
        of three heavy elements.

        Args:
            which_Z1 (str): name of the first heavy element.
                Current options are "h2o", "sio2", "fe" or "co".
            Z1 (float): mass-fraction of the first heavy element.
            which_Z2 (str): name of the second heavy element.
                Current options are "h2o", "sio2", "fe" or "co".
            Z2 (float): mass-fraction of the second heavy element.
            which_Z3 (str): name of the third heavy element.
                Current options are "h2o", "sio2", "fe" or "co".
            Z3 (float): mass-fraction of the third heavy element.
            store_table (bool, optional): whether to store the table.
                Defaults to False.

        Raises:
            ValueError: raised if mass fractions don't sum to one.

        Returns:
            NDArray: table of the mixture.
                Columns are: logT, logP, logRho, logU, and logS.
        """
        if not np.isclose(Z1 + Z2 + Z3, 1, atol=1e-5):
            msg = "Mass fractions of the heavy elements must sum to one."
            raise ValueError(msg)
        Z1_table = self.__load_z_pt_table(which_Z1)
        Z2_table = self.__load_z_pt_table(which_Z2)
        Z3_table = self.__load_z_pt_table(which_Z3)

        logT = Z1_table[:, 0]
        logP = Z1_table[:, 1]
        logRho_Z1 = Z1_table[:, 2]
        logU_Z1 = Z1_table[:, 3]
        logS_Z1 = Z1_table[:, 4]

        logRho_Z2 = Z2_table[:, 2]
        logU_Z2 = Z2_table[:, 3]
        logS_Z2 = Z2_table[:, 4]

        logRho_Z3 = Z3_table[:, 2]
        logU_Z3 = Z3_table[:, 3]
        logS_Z3 = Z3_table[:, 4]

        logRho = -np.log10(
            Z1 / np.power(10, logRho_Z1)
            + Z2 / np.power(10, logRho_Z2)
            + Z3 / np.power(10, logRho_Z3)
        )
        logU = np.log10(
            Z1 * np.power(10, logU_Z1)
            + Z2 * np.power(10, logU_Z2)
            + Z3 * np.power(10, logU_Z3)
        )
        logS = np.log10(
            Z1 * np.power(10, logS_Z1)
            + Z2 * np.power(10, logS_Z2)
            + Z3 * np.power(10, logS_Z3)
        )
        grad_ad = 0.3 * np.ones_like(logRho)  # dummy values
        Z_table = np.column_stack([logT, logP, logRho, logU, logS, grad_ad])

        if store_table:
            Z1 = 100 * Z1
            Z2 = 100 * Z2
            Z3 = 100 * Z3
            fname = f"qeos_pt_{which_Z1}_{Z1:02.0f}"
            fname = fname + f"_{which_Z2}_{Z2:02.0f}"
            fname = fname + f"_{which_Z3}_{Z3:02.0f}.data"
            dst = os.path.join(self.tables_path, fname)
            np.savetxt(dst, Z_table, header=self.z_pt_header, fmt="%.8e")
        return Z_table

    @staticmethod
    def smooth_xy_table(
        table: NDArray, which_y: str, num_smoothing_rounds: int = 1
    ) -> NDArray:
        """Smoothes the CMS hydrogen-helium tables by taking
        the average of the original point and the four nearest neighbours
        on the two-dimensional grid.

        Args:
            table (NDArray): original unsmoothed table
            which_y (str): defines the second independent variable
                for the equation of state (logT, which_y)
            num_smoothing_rounds (int, optional): how many times the table
                is smoothed. Defaults to 1.

        Returns:
            NDArray: smoothed table
        """
        i_y = 1 if which_y == "logP" else 2

        input_ndim = table.ndim
        if input_ndim == 2:
            num_xs = np.unique(table[:, 0]).size
            num_ys = np.unique(table[:, i_y]).size
            table = table.reshape((num_xs, num_ys, table.shape[1]))
        else:
            num_xs = table.shape[0]
            num_ys = table.shape[1]

        def do_smooth_table(in_table, num_xs, num_ys):
            out_table = np.copy(in_table)
            for i in range(num_xs):
                if i < 1:
                    continue
                elif i > num_xs - 2:
                    break
                for j in range(num_ys):
                    if j < 1:
                        continue
                    elif j > num_ys - 2:
                        break
                    out_table[i, j, 2:] = (
                        in_table[i, j, 2:]
                        + in_table[i + 1, j, 2:]
                        + in_table[i - 1, j, 2:]
                        + in_table[i, j - 1, 2:]
                        + in_table[i, j + 1, 2:]
                    ) / 5
            return out_table

        in_table = np.copy(table)
        if which_y == "logRho" and input_ndim == 2:
            # switch logP and logRho columns
            in_table[:, 1] = table[:, 2]
            in_table[:, 2] = table[:, 1]
        for _ in range(num_smoothing_rounds):
            out_table = do_smooth_table(in_table, num_xs, num_ys)
            in_table = out_table
        if which_y == "logRho" and input_ndim == 2:
            # switch logP and logRho columns again
            out_table[:, 1] = in_table[:, 2]
            out_table[:, 2] = in_table[:, 1]

        return out_table

    @staticmethod
    def smooth_z_table(table: NDArray, num_smoothing_rounds: int = 1) -> NDArray:
        """Smoothes the qeos heavy-element table by taking
        the average of the original point and the four nearest neighbours
        on the two-dimensional grid. Works for both (logT, logRho) and
        (logT, logP) tables.

        Args:
            table (NDArray): original unsmoothed table
            num_smoothing_rounds (int, optional): how many times the table
                is smoothed. Defaults to 1.

        Returns:
            NDArray: smoothed table
        """
        input_ndim = table.ndim
        if input_ndim == 2:
            num_xs = np.unique(table[:, 0]).size
            num_ys = np.unique(table[:, 1]).size
            table = table.reshape((num_xs, num_ys, table.shape[1]))
        else:
            num_xs = table.shape[0]
            num_ys = table.shape[1]

        def do_smooth_table(in_table, num_xs, num_ys):
            out_table = np.copy(in_table)
            for i in range(num_xs):
                if i < 1:
                    continue
                elif i > num_xs - 2:
                    break
                for j in range(num_ys):
                    if j < 1:
                        continue
                    elif j > num_ys - 2:
                        break
                    out_table[i, j, 2:] = (
                        in_table[i, j, 2:]
                        + in_table[i + 1, j, 2:]
                        + in_table[i - 1, j, 2:]
                        + in_table[i, j - 1, 2:]
                        + in_table[i, j + 1, 2:]
                    ) / 5
            return out_table

        in_table = np.copy(table)
        for _ in range(num_smoothing_rounds):
            out_table = do_smooth_table(in_table, num_xs, num_ys)
            in_table = out_table

        if input_ndim == 2:
            out_table = out_table.reshape((-1, table.shape[2]))
        return out_table

    @staticmethod
    def make_monotonic(data: ArrayLike, element: str, DT: bool) -> NDArray:
        """Makes the data monotonic with respect to logT.

        Args:
            data (ArrayLike): The input data.
                element (str): Which table to load.
            DT (bool): whether to do (logRho, logT) or (logP, logT)

        Returns:
            NDArray: Monotonic version of the data.
        """
        if element in ["hydrogen", "helium"]:
            if DT:
                y = data[:, 2]  # logRho
                d = {"logP": 1, "logU": 3, "logS": 4}
            else:
                y = data[:, 1]  # logP
                d = {"logRho": 2, "logU": 3, "logS": 4}
        else:
            y = data[:, 1]  # logRho or logP
            if DT:
                d = {"logP": 2, "logU": 3, "logS": 4}
            else:
                d = {"logRho": 2, "logU": 3, "logS": 4}

        for i in d.values():
            z = data[:, i]
            for u in np.unique(y):
                idcs = np.where(y == u)
                zi = z[idcs]
                check_grad = np.gradient(zi) < 0
                if np.any(check_grad):
                    jdcs = np.where(check_grad)
                    max_accum = np.maximum.accumulate(zi)
                    zi[jdcs] = max_accum[jdcs]
                z[idcs] = zi
            data[:, i] = z

        return data


if __name__ == "__main__":
    T = TableLoader(which_hhe="cms")
    # convert effective hydrogen table
    # from (logT, logP) to (logT, logRho)
    T.invert_xeff_pt_table(
        kind="linear",
        extrapolate=True,
        smooth_table=False,
        num_smoothing_rounds=2,
        store_table=True,
    )

    # convert h2o, sio2 and fe tables from
    # (logT, logRho) to (logT, logP)
    for element in ["h2o", "sio2", "fe", "co"]:
        T.invert_z_table(
            which_variables="pt",
            which_heavy=element,
            kind="pchip",
            extrapolate=True,
            smooth_table=True,
            num_smoothing_rounds=1,
            store_table=True,
        )

    # create a 50-50 h2o-sio2 mixture pt table
    T.mix_heavy_elements(
        which_Z1="h2o",
        Z1=0.5,
        which_Z2="sio2",
        Z2=0.5,
        which_Z3="fe",
        Z3=0,
        store_table=True,
    )

    # create a 67-33 h2o-sio2 mixture pt table
    T = TableLoader(which_heavy="mixture", Z1=0.67, Z2=0.33, Z3=0)

    # create smoothed dt tables
    num_smoothing_rounds = 2
    for element in ["h2o", "aqua", "sio2", "fe", "co", "mixture"]:
        T = TableLoader(which_heavy=element)
        table = T.z_DT_table
        smoothed_table = T.smooth_z_table(
            table, num_smoothing_rounds=num_smoothing_rounds
        )
        if element == "mixture":
            element = "h2o_50_sio2_50_fe_00"
        fname = f"qeos_smoothed_dt_{element}.data"
        dst = os.path.join(T.tables_path, fname)
        np.savetxt(dst, smoothed_table, fmt="%.8e", header=T.z_dt_header)

    # create smoothed pt tables
    for element in ["h2o", "aqua", "sio2", "fe", "co", "mixture"]:
        T = TableLoader(which_heavy=element)
        table = T.z_PT_table
        smoothed_table = T.smooth_z_table(
            table, num_smoothing_rounds=num_smoothing_rounds
        )
        if element == "mixture":
            element = "h2o_50_sio2_50_fe_00"
        if element == "aqua":
            fname = "aqua_smoothed_pt_h2o.data"
        else:
            fname = f"qeos_smoothed_pt_{element}.data"
        dst = os.path.join(T.tables_path, fname)
        np.savetxt(dst, smoothed_table, fmt="%.8e", header=T.z_dt_header)
