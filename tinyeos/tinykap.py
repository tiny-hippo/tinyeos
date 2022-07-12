import os
import pickle
import numpy as np
from typing import Tuple
from pathlib import Path
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d, RegularGridInterpolator
from tinyeos.support import Z_sun, get_FeH_from_Z, get_Z_from_FeH


class TinyFreedmanKap:
    """Temperature-pressure opacity for a gaseous mixture from
    Freedman et al. (2014). Units are cgs everywhere.
    """

    def __init__(
        self, use_fit: bool = False, build_interpolants: bool = False
    ) -> None:
        """__init__ method. Loads the interpolant files from the disk.
        Optionally, it re-builds the interpolants from tables.

        Args:
            use_analytical_fit (bool, optional): whether to use the analytical
                fit for the opacity calculation. Defaults to False.
            build_interpolants (bool, optional): whether to build interpolants
                from the tables. Defaults to False.
        """
        self.FeHs = np.array([0, 0.5, 0.7, 1.0, 1.5, 1.7])
        self.num_FeHs = self.FeHs.size
        self.tables_path = Path(__file__).parent / "data/kap/tables"
        self.cache_path = Path(__file__).parent / "data/kap/interpolants"

        if use_fit:
            self.func = self.evaluate_fit
        else:
            self.func = self.evaluate

        if build_interpolants:
            for FeH in self.FeHs:
                Z = get_Z_from_FeH(FeH)
                self.__cache_freedman_kap_interp(Z)

        self.rgis = []
        for FeH in self.FeHs:
            Z = get_Z_from_FeH(FeH)
            rgi = self.__load_freedman_kap_interp(Z)
            self.rgis.append(rgi)

    def __call__(self, logT: float, logP: float, Z: float) -> float:
        """__call__ method acting as convenience wrapper for the evaluate
        method. Calculates the Freedman opacity the gaseous mixture.

        Args:
            logT (float): log10 of the temperature.
            logP (float): log10 of the pressure.
            Z (float): heavy-element mass fraction.

        Returns:
            float: opacity
        """
        return self.func(logT, logP, Z)

    @staticmethod
    def __get_freedman_kap_fname(Z: float) -> str:
        """Returns the filename of the Freedman opacity
        table for a given heavy-element mass fraction.

        Args:
            Z (float): heavy-element mass fraction

        Returns:
            str: table filename
        """
        FeH = get_FeH_from_Z(Z)
        if FeH >= 0 and FeH < 0.5:
            suffix = "0.0"
        elif FeH >= 0.5 and FeH < 0.7:
            suffix = "0.5"
        elif FeH >= 0.7 and FeH < 1.0:
            suffix = "0.7"
        elif FeH >= 1.0 and FeH < 1.5:
            suffix = "1.0"
        elif FeH >= 1.5 and FeH < 1.7:
            suffix = "1.5"
        elif FeH >= 1.7:
            suffix = "1.7"
        fname = f"freedman_{suffix}.txt"
        return fname

    def __get_freedman_kap_tables(self, Z: float) -> Tuple:
        """Loads the Freedman opacity table for
        a given heavy-element mass fraction.

        Args:
            Z (float): heavy-element mass fraction

        Returns:
            Tuple: temperature, pressure and opacity
        """
        i_T = 0
        i_P = 1
        i_kap = 3
        fname = self.__get_freedman_kap_fname(Z)
        src = os.path.join(self.tables_path, fname)
        with open(src, "r") as file:
            data = np.loadtxt(file, skiprows=38, dtype=np.float64)
        data = np.transpose(data)
        return (data[i_T, :], data[i_P, :], data[i_kap, :])

    def __get_freedman_kap_from_tables(
        self, T: float, P: float, Z: float, num_Ts: int = 4, kind: str = "linear"
    ) -> float:
        """Calculates the Freedman opacity for a given temperature,
        pressure and heavy-element mass fraction directly from the table.

        Args:
            T (float): temperature
            P (float): pressure
            Z (float): heavy-element mass fraction
            num_Ts (int, optional): number of temperature points for the
                interpolation. Defaults to 4.
            kind (str, optional): interpolation method. Defaults to "linear".

        Raises:
            ValueError: temperature out of range

        Returns:
            float: opacity
        """
        Ts, Ps, kappas = self.__get_freedman_kap_tables(Z)
        unique_Ts = np.unique(Ts)
        T_min = np.min(unique_Ts)
        T_max = np.max(unique_Ts)
        if T < T_min:
            msg = f"T = {T} < T_min = {T_max}"
            raise ValueError(msg)
        elif T > T_max:
            msg = f"T = {T} > T_max = {T_max}"
            raise ValueError(msg)

        i = (np.abs(unique_Ts - T)).argmin()
        i_start = int(i - num_Ts / 2)
        while i_start < 0:
            i_start = i_start + 1
        if i_start == i:
            i_end = i_start + num_Ts
        else:
            i_end = int(i + num_Ts / 2 + (i - i_start))
        if i_end == i:
            i_start = i - 4

        new_Ts = unique_Ts[i_start : i_end + 1]
        new_kappas = np.zeros_like(new_Ts)
        for i, new_T in enumerate(new_Ts):
            in_range = True
            idcs = np.where(Ts == new_T)
            P_min = np.min(Ps[idcs])
            P_max = np.max(Ps[idcs])
            if P < P_min:
                msg = f"P = {P} < P_min = {P_min} for T = {new_T}"
                in_range = False
            elif P > P_max:
                msg = f"P = {P} > P_max = {P_max} for T = {new_T}"
                in_range = False
            if in_range:
                f = interp1d(
                    Ps[idcs], kappas[idcs], kind=kind, fill_value="extrapolate"
                )
                new_kappas[i] = f(P)
            else:
                new_kappas[i] = np.nan

        if kind == "linear":
            min_entries = 2
        elif kind == "cubic":
            min_entries = 3
        num_nans = np.sum(np.isnan(new_kappas))
        if num_Ts + 1 - num_nans < min_entries:
            msg = "failed for ({T}, {P})"
            return np.nan
        if num_nans > 0:
            idcs = np.isfinite(new_kappas)
            new_kappas = new_kappas[idcs]
            new_Ts = new_Ts[idcs]
        f = interp1d(new_Ts, new_kappas, kind=kind, fill_value="extrapolate")
        return float(f(T))

    def __get_freedman_kap_interp(self, Z: float) -> RegularGridInterpolator:
        """Builds a RegularGridInterpolator for a given heavy-element
        mass fraction from the Freedman opacity tables.

        Args:
            Z (float): heavy-element mass fraction

        Raises:
            ValueError: raised if the interpolation returns
                any nans.

        Returns:
            RegularGridInterpolator: temperature-pressure interpolation
                for the Freedman opacity.
        """
        Ts, Ps, _ = self.__get_freedman_kap_tables(Z)
        unique_Ts = np.unique(Ts)
        unique_Ps = np.unique(Ps)

        num_Ts = unique_Ts.size
        num_Ps = unique_Ps.size
        kap = np.zeros((num_Ts, num_Ps))
        for i, T in enumerate(unique_Ts):
            for j, P in enumerate(unique_Ps):
                kap[i, j] = self.__get_freedman_kap_from_tables(
                    T, P, Z, num_Ts=4, kind="linear"
                )

        kap_fixed = np.copy(kap)
        if np.any(np.isnan(kap)):
            fn = RegularGridInterpolator((unique_Ts, unique_Ps), kap, method="nearest")
            idcs = np.where(np.isnan(kap))
            i_nan = idcs[0]
            j_nan = idcs[1]
            kap_fixed = np.copy(kap)
            for k in range(i_nan.size):
                i = i_nan[k]
                j = j_nan[k]
                T = unique_Ts[i]
                P = unique_Ps[j - 1]
                kap_fixed[i, j] = fn((T, P))
            if np.any(np.isnan(kap_fixed)):
                raise ValueError()

        rgi = RegularGridInterpolator(
            (unique_Ts, unique_Ps), kap_fixed, method="linear"
        )
        # rbs = RectBivariateSpline(unique_Ts, unique_Ps, kap_fixed, kx=1, ky=1)
        return rgi

    def __cache_freedman_kap_interp(self, Z: float) -> None:
        """Caches the RegularGridInterpolator object of the
        Freedman opacity for a given heavy-element mass fraction.

        Args:
            Z (float): heavy-element mass fraction
        """
        fname = self.__get_freedman_kap_fname(Z)
        fname = fname.replace("txt", "pkl")
        dst = os.path.join(self.cache_path, fname)
        rgi = self.__get_freedman_kap_interp(Z)
        with open(dst, "wb") as file:
            pickle.dump(rgi, file)

    def __load_freedman_kap_interp(self, Z: float) -> RegularGridInterpolator:
        """Loads the RegularGridInterpolator object of the
        Freedman opacity for a given heavz/element mass fraction.

        Args:
            Z (float): heavy-element mass fraction.

        Raises:
            FileNotFoundError: missing interpolant cache

        Returns:
            RegularGridInterpolator: _description_
        """
        fname = self.__get_freedman_kap_fname(Z)
        fname = fname.replace("txt", "pkl")
        src = os.path.join(self.cache_path, fname)
        if not os.path.isfile(src):
            msg = f"missing interpolant cache {src} - try to re-build"
            raise FileNotFoundError(msg)
        with open(src, "rb") as file:
            rgi = pickle.load(file)
        return rgi

    def evaluate(self, logT: float, logP: float, Z: float) -> float:
        """Calculates the Freedman opacity of the gaseous mixture.

        Args:
            logT (float): log10 of the temperature.
            logP (float): log10 of the pressure.
            Z (float): heavy-element mass fraction.

        Returns:
            float: opacity
        """
        T = 10**logT
        P = 10**logP
        FeH = get_FeH_from_Z(Z)
        if FeH > np.max(self.FeHs):
            i = self.num_FeHs - 2
        else:
            i = (np.abs(self.FeHs - FeH)).argmin()
        j = i + 1

        rgi0 = self.rgis[i]
        rgi1 = self.rgis[j]
        Z0 = get_Z_from_FeH(self.FeHs[i])
        Z1 = get_Z_from_FeH(self.FeHs[j])
        kap0 = rgi0((T, P))
        kap1 = rgi1((T, P))
        kap = kap0 + (Z - Z0) * (kap1 - kap0) / (Z1 - Z0)
        return kap

    def evaluate_fit(self, logT: ArrayLike, logP: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """Calculates the Freedman opacity of a gaseous mixture with
        the analytic fit (see eqs. 3-5).

        Args:
            logT (ArrayLike): log10 of the temperature
            logP (ArrayLike): log10 of the pressure
            Z (ArrayLike): heavy-element mass fract

        Returns:
            ArrayLike: opacity
        """

        if not isinstance(logT, np.ndarray):
            logT = np.array(logT)
        if not isinstance(logP, np.ndarray):
            logP = np.array(logP)

        if not logT.shape == logP.shape:
            msg = "logT and logP must have equal shape"
            raise ValueError(msg)

        T = np.power(10, logT)
        P = np.power(10, logP)
        met = np.log10(Z / Z_sun)

        input_ndim = logT.ndim
        # coefficients used for opacity fit (table 2)
        if input_ndim > 0:
            eyes = np.ones_like(T)
            c1 = 10.602 * eyes
            c2 = 2.882 * eyes
            c3 = 6.09e-15 * eyes
            c4 = 2.954 * eyes
            c5 = -2.256 * eyes
            c6 = 0.843 * eyes
            c7 = -5.490 * eyes

            c8 = np.zeros_like(T)
            c9 = np.zeros_like(T)
            c10 = np.zeros_like(T)
            c11 = np.zeros_like(T)
            c12 = np.zeros_like(T)
            c13 = np.zeros_like(T)

            i = T < 800
            c8[i] = -14.051
            c9[i] = 3.055
            c10[i] = 0.024
            c11[i] = 1.877
            c12[i] = -0.445
            c13[i] = 0.8321

            i = ~i
            c8[i] = 82.241
            c9[i] = -55.456
            c10[i] = 8.754
            c11[i] = 0.7048
            c12[i] = -0.0414
            c13[i] = 0.8321

        else:
            c1 = 10.602
            c2 = 2.882
            c3 = 6.09e-15
            c4 = 2.954
            c5 = -2.256
            c6 = 0.843
            c7 = -5.490
            if T < 800:
                c8 = -14.051
                c9 = 3.055
                c10 = 0.024
                c11 = 1.877
                c12 = -0.445
                c13 = 0.8321
            else:
                c8 = 82.241
                c9 = -55.456
                c10 = 8.754
                c11 = 0.7048
                c12 = -0.0414
                c13 = 0.8321

        # eq. 4
        log10_kap_lowP = (
            c1 * np.arctan(logT - c2)
            - c3 / (logP + c4) * np.exp(np.power(logT - c5, 2))
            + c6 * met
            + c7
        )

        # eq. 5
        log10_kap_highP = (
            c8
            + c9 * logT
            + c10 * np.power(logT, 2)
            + logP * (c11 + c12 * logT)
            + c13 * met * (0.5 + 1 / np.pi * np.arctan((logT - 2.5) / 0.2))
        )

        kap_gas = np.power(10, log10_kap_lowP) + np.power(10, log10_kap_highP)
        return kap_gas
