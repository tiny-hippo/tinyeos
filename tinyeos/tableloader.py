import os
import pickle
import numpy as np
from numpy.typing import ArrayLike, NDArray
from pathlib import Path


class TableLoader:
    """ Loads the equation of state tables for hydrogen, helium
    and a heavy element.

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
        logRho_max (float): max. density
        logRho_min (float): min. density
        logT_max (float): max. temperature
        logT_min (float): min. temperature
        x_DT_table (ndarray): hydrogen (logRho, logT) table
        x_PT_table (ndarray): hydrogen (logP, logT) table
        y_DT_table (ndarray): helium (logRho, logT) table
        y_PT_table (ndarray): helium (logP, logT) table
        z_DT_table (ndarray): heavy-element (logRho, logT) table
        z_PT_table (ndarray): heavy-element (logP, logT) table
    """
    def __init__(self, which_heavy: str = "water", which_hhe: str = "cms") -> None:
        """ _init__ method. Sets the equation of state
        boundaries and loads the tables.

        Args:
            which_heavy (str, optional): Which heavy-element equation of state
            to use. Defaults to "water".
            which_hhe (str, optional): Which hydrogen-helium equation of state
            to use. Defaults to "cms".
        """

        self.table_path = Path(__file__).parent / "data/tables"
        self.logRho_max = 2.00
        self.logRho_min = -8.00
        self.logT_max = 6.00
        self.logT_min = 2.00

        self.load_xy_DT_tables(which_hhe)
        self.load_xy_PT_tables(which_hhe)
        self.load_z_DT_table(which_heavy)
        self.load_z_PT_table(which_heavy)

    def load_xy_DT_tables(self, which_hhe: str = "cms") -> None:
        """ Loads the hydrogen and helium (logRho, logT) tables.

        Args:
            which_hhe (str, optional): Which hydrogen-helium equation of state
            to use. Defaults to "cms".

        Raises:
            NotImplementedError: Raised if which_hhe option is unavailable.
        """

        if which_hhe == "cms":
            src = os.path.join(self.table_path, "hydrogen_DT_table.pkl")
        elif which_hhe == "scvh":
            # to-do: test extended scvh dt tables
            src = os.path.join(self.table_path,
                               "hydrogen_scvh_extended_DT_table.pkl")
        else:
            raise NotImplementedError("This table is not available.")
        with open(src, "rb") as file:
            data = pickle.load(file)
        # columns = ["logT", "logP", "logRho", "logU", "logS", "dlnRho/dlnT",
        #          "dlnRho/dlnP", "dlnS/dlnT", "dlnS/dlnP", "grad_ad",
        #          "log_free_e", "mu"]
        self.x_DT_table = data

        if which_hhe == "cms":
            src = os.path.join(self.table_path, "helium_DT_table.pkl")
        elif which_hhe == "scvh":
            # to-do: test extended scvh dt tables
            src = os.path.join(self.table_path,
                               "helium_scvh_extended_DT_table.pkl")
        with open(src, "rb") as file:
            data = pickle.load(file)
        self.y_DT_table = data

    def load_xy_PT_tables(self, which_hhe: str = "cms") -> None:
        """ Loads the hydrogen and helium (logP, logT) tables.

        Args:
            which_hhe (str, optional):  Which hydrogen-helium equation of state
            to use. Defaults to "cms".

        Raises:
            NotImplementedError: Raised if which_hhe option is unvailable.
        """

        if which_hhe == "cms":
            src = os.path.join(self.table_path, "hydrogen_PT_table.pkl")
        elif which_hhe == "scvh":
            src = os.path.join(self.table_path, "hydrogen_scvh_PT_table.pkl")
        else:
            raise NotImplementedError("This table is not available.")
        with open(src, "rb") as file:
            data = pickle.load(file)
        # columns = ["logT", "logP", "logRho", "logU", "logS", "dlnRho/dlnT",
        #          "dlnRho/dlnP", "dlnS/dlnT", "dlnS/dlnP", "grad_ad",
        #          "log_free_e", mu]
        self.x_PT_table = data

        if which_hhe == "cms":
            src = os.path.join(self.table_path, "helium_PT_table.pkl")
        elif which_hhe == "scvh":
            src = os.path.join(self.table_path, "helium_scvh_PT_table.pkl")
        with open(src, "rb") as file:
            data = pickle.load(file)
        self.y_PT_table = data

    def load_z_DT_table(self, which_heavy: str) -> None:
        """ Loads the heavy-element (logRho, logT) tables.

        Args:
            which_heavy (str): Which heavy-element equation of state to use.

        Raises:
            NotImplementedError: Raised if which_heavy option is unavailable.
        """
        if which_heavy == "water":
            # fname = "qeos_h2o_dt_cgs.data"
            fname = "qeos_h2o_dt_cgs_smoothed.data"
        elif which_heavy == "aqua":
            fname = "aqua_dt_cgs.data"
        elif which_heavy == "rock":
            fname = "qeos_sio2_dt_cgs.data"
        elif which_heavy == "iron":
            fname = "qeos_fe_dt_cgs.data"
        elif which_heavy == "mixture":
            fname = "qeos_mix_dt_cgs.data"
        else:
            raise NotImplementedError("This heavy element is not available")

        src = os.path.join(self.table_path, fname)
        data = np.loadtxt(src, skiprows=1, dtype=np.float64)
        data = np.loadtxt(src, skiprows=1, dtype=np.float64)
        data = data[np.where(data[:, 0] > 1.90)]
        # columns = ["logT", "logRho", "logP", "logU", "logS", "grad_ad"]
        self.z_DT_table = data

    def load_z_PT_table(self, which_heavy: str):
        """ Loads the heavy-element (logP, logT) tables.

        Args:
            which_heavy (str): Which heavy-element equation of state to use.

        Raises:
            NotImplementedError: Raised if which_heavy option is unavailable.
        """
        if which_heavy == "water":
            fname = "qeos_h2o_pt_cgs.data"
        elif which_heavy == "aqua":
            fname = "aqua_pt_cgs.data"
        elif which_heavy == "rock":
            fname = "qeos_sio2_pt_cgs.data"
        elif which_heavy == "mixture":
            fname = "qeos_mix_pt_cgs.data"
        # elif (which_heavy == "iron"):
        #     fname = "fe_PT_table.data"
        else:
            raise NotImplementedError("This heavy element is not available")

        src = os.path.join(self.table_path, fname)
        data = np.loadtxt(src, skiprows=1, dtype=np.float64)
        data = data[np.where(data[:, 0] > 1.90)]
        # columns = ["logT", "logP", "logRho", "logU", "logS", "grad_ad"]
        self.z_PT_table = data

    @staticmethod
    def make_monotonic(data: ArrayLike, element: str, DT: bool) -> NDArray:
        """ Makes the data monotonic with respect to logT.

        Args:
            data (ArrayLike): The input data.
            element (str): Which table to load.
            DT (bool): whether to do (logRho, logT) or (logP, logT)

        Returns:
            NDArray: Monotonic version of the data.
        """

        x = data[:, 0]  # logT
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
