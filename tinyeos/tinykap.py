import os
import pickle
from pathlib import Path

import numpy as np
import requests
from numpy.typing import ArrayLike
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, interp1d

from tinyeos.support import Z_sun, get_FeH_from_Z, get_Z_from_FeH, sigma_b


class TinyOpacity:
    def __init__(
        self,
        build_interpolants: bool = False,
        use_kap_rad_fit: bool = False,
        reload_kap_ec_tables: bool = False,
    ) -> None:
        self.tfk = TinyFreedmanOpacity(
            use_fit=use_kap_rad_fit, build_interpolants=build_interpolants
        )
        self.tec = TinyElectronConduction(
            build_interpolants=build_interpolants, reload_table=reload_kap_ec_tables
        )

        self.use_kap_rad_fit = use_kap_rad_fit

    def __call__(
        self,
        logT: ArrayLike,
        logRho: ArrayLike,
        logP: ArrayLike,
        Z: ArrayLike,
        logz: ArrayLike,
        add_grain_opacity: bool = False,
        f_grains: float = 1.0,
    ) -> ArrayLike:
        return self.evaluate(
            logT=logT,
            logRho=logRho,
            logP=logP,
            Z=Z,
            logz=logz,
            add_grain_opacity=add_grain_opacity,
            f_grains=f_grains,
        )

    def __get_ism_grain_opacity(
        self, logT: float, logRho: float, f_grains: float
    ) -> float:
        """Calculates the grain opacity for the ISM following the fit
        from Valencia et al. (2013).

        Args:
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            f_grains (float): grain opacity factor.

        Returns:
            float: grain opacity.
        """
        T_6 = 10**logT / 1e6
        logR_bar = np.log10(10**logRho / T_6**3)
        logT1_star = 0.0245 * logR_bar + 3.096
        logT2_star = 0.0245 * logR_bar + 3.221
        if logT > logT2_star:
            kap_grains = 0
        elif logT < logT1_star:
            kap_grains = self.__get_grain_opacity(logT=logT, f_grains=f_grains)
        else:
            # linear interpolation
            kap_grains_T1 = self.__get_grain_opacity(logT=logT1_star, f_grains=f_grains)
            kap_grains_T2 = 0
            kap_grains = kap_grains_T1 + (kap_grains_T2 - kap_grains_T1) / (
                logT2_star - logT1_star
            ) * (logT - logT1_star)
        return kap_grains

    @staticmethod
    def __get_grain_opacity(logT: ArrayLike, f_grains: ArrayLike) -> ArrayLike:
        log_kap_grains = 0.430 + 1.3143 * (logT - 2.85)
        kap_grains = f_grains * 10**log_kap_grains
        return kap_grains

    def evaluate_kap_rad(self, logT: float, logP: float, Z: float) -> float:
        return self.tfk(logT, logP, Z)

    def evaluate_kap_ec(
        self, logT: ArrayLike, logRho: ArrayLike, logz: ArrayLike
    ) -> ArrayLike:
        return self.tec(logT, logRho, logz)

    def evaluate(
        self,
        logT: ArrayLike,
        logRho: ArrayLike,
        logP: ArrayLike,
        Z: ArrayLike,
        logz: ArrayLike,
        add_grain_opacity: bool = False,
        f_grains: float = 1.0,
    ) -> ArrayLike:
        if not self.use_kap_rad_fit:
            kap_rad_func = np.vectorize(self.evaluate_kap_rad)
            kap_rad = kap_rad_func(logT, logP, Z)
        else:
            kap_rad = self.evaluate_kap_rad(logT, logP, Z)
        if add_grain_opacity:
            kap_grains = self.__get_ism_grain_opacity(
                logT=logT, logRho=logRho, f_grains=f_grains
            )
            kap_rad = kap_rad + kap_grains
        if logT < 3 or logRho < -6:
            # no electron conduction in this regime
            kap_ec = 1e99
        else:
            kap_ec = self.evaluate_kap_ec(logT, logRho, logz)
        # combine radiative and conductive opacities
        kap = 1 / (1 / kap_rad + 1 / kap_ec)
        return kap


class TinyFreedmanOpacity:
    """Temperature-pressure opacity for a gaseous mixture from
    Freedman et al. (2014). Units are cgs everywhere.
    """

    def __init__(self, use_fit: bool = False, build_interpolants: bool = False) -> None:
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

    # def __get_rgi(self):
    #     num_logTs = 128
    #     num_logPs = 20
    #     min_logT = np.log10(75)
    #     max_logT = np.log10(4000)
    #     min_logP = 0  # 1 barye or 1e-6 bar
    #     max_logP = 8

    #     x = np.linspace(min_logT, max_logT, num_logTs)
    #     y = np.linspace(min_logP, max_logP, num_logPs)
    #     z = np.array([0, 0.5, 0.7, 1, 1.5, 1.7])
    #     z = get_Z_from_FeH(z)
    #     data = np.zeros((num_logTs, num_logPs, z.size))
    #     for i in range(self.num_logzs):
    #         data[:, :, i] = self.rbs[i](x, y, grid=True)
    #     rgi = RegularGridInterpolator((x, y, z), data, method="linear")
    #     return rgi

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

    def __get_freedman_kap_tables(self, Z: float) -> tuple:
        """Loads the Freedman opacity table for
        a given heavy-element mass fraction.

        Args:
            Z (float): heavy-element mass fraction

        Returns:
            tuple: temperature, pressure and opacity
        """
        i_T = 0
        i_P = 1
        i_kap = 3
        fname = self.__get_freedman_kap_fname(Z)
        src = os.path.join(self.tables_path, fname)
        with open(src) as file:
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
        if T_min > T:
            msg = f"T = {T} < T_min = {T_max}"
            raise ValueError(msg)
        elif T_max < T:
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
            if P_min > P:
                msg = f"P = {P} < P_min = {P_min} for T = {new_T}"
                in_range = False
            elif P_max < P:
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
            (unique_Ts, unique_Ps),
            kap_fixed,
            method="pchip",
            bounds_error=False,
            fill_value=np.nan,
        )
        # rbs = RectBivariateSpline(unique_Ts, unique_Ps, kap_fixed, kx=1, ky=1)
        # rgi = rbs
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
            Z (ArrayLike): heavy-element mass fraction

        Returns:
            ArrayLike: opacity
        """

        if not isinstance(logT, np.ndarray):
            logT = np.array(logT)
        if not isinstance(logP, np.ndarray):
            logP = np.array(logP)

        if logT.shape != logP.shape:
            msg = "logT and logP must have equal shape"
            raise ValueError(msg)

        T = np.power(10, logT)
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


class TinyElectronConduction:
    """Temperature-density (electron) conductive opacity for stellar plasmas
    from Potekhin (http://www.ioffe.ru/astro/conduct/condint.html).
    Units are cgs everywhere.
    """

    def __init__(
        self,
        reload_table: bool = False,
        build_interpolants: bool = False,
        download_tables: bool = False,
    ) -> None:
        """

        Args:
            reload_table (bool, optional): whether to reload the conductivity
                tables instead of using the cache. Defaults to False.
            build_interpolants (bool, optional): whether to build the
                interpolants instead of using the cache.
                Defaults to False.
            download_tables (bool, optional): whether to download the
                Potekhin (2021) tables from the web. Defaults to False.
        """
        self.num_logTs = 29
        self.num_logRhos = 71
        self.num_logzs = 15
        self.i_logT = 0
        self.i_logRho = 1
        self.i_logz = 2  # ion charge number
        self.i_logK = 3  # thermal conductivity

        self.table_data_path = Path(__file__).parent / "data/kap/tables"
        self.table_cache_path = Path(__file__).parent / "data/kap/tables/cache"
        self.interp_cache_path = Path(__file__).parent / "data/kap/interpolants"
        self.table_path = os.path.join(self.table_data_path, "condtabl.data")
        self.cache_path = os.path.join(self.table_cache_path, "cond_data.pkl")
        self.rbs_path = os.path.join(self.interp_cache_path, "cond_rbs.pkl")
        self.rgi_path = os.path.join(self.interp_cache_path, "cond_rgi.pkl")

        if download_tables:
            page_url = "http://www.ioffe.ru/astro/conduct/"
            table_name = "condtabXXX.dat"
            names = ["21wd", "21_I", "21nd", "21sd"]
            for name in names:
                current_table = table_name.replace("XXX", name)
                file_url = page_url + current_table
                response = requests.get(file_url)
                dst = os.path.join(self.table_data_path, current_table)
                with open(dst, "wb") as file:
                    file.write(response.content)

        if not os.path.exists(self.cache_path) or reload_table:
            self.table = self.__get_potekhin_table()
            with open(self.cache_path, "wb") as file:
                pickle.dump(self.table, file)
        else:
            with open(self.cache_path, "rb") as file:
                self.table = pickle.load(file)

        self.logTs = np.unique(self.table[0, :, self.i_logT])
        self.logRhos = np.unique(self.table[0, :, self.i_logRho])
        self.logzs = self.table[:, 0, self.i_logz]

        if not os.path.exists(self.rbs_path) or build_interpolants:
            self.num_logzs = 15
            self.rbs = []
            for i in range(self.num_logzs):
                self.rbs.append(self.__get_rbs(i))
            with open(self.rbs_path, "wb") as file:
                pickle.dump(self.rbs, file)
        else:
            with open(self.rbs_path, "rb") as file:
                self.rbs = pickle.load(file)

        if not os.path.exists(self.rgi_path) or build_interpolants:
            self.rgi = self.__get_rgi()
            with open(self.rgi_path, "wb") as file:
                pickle.dump(self.rgi, file)
        else:
            with open(self.rgi_path, "rb") as file:
                self.rgi = pickle.load(file)

    def __call__(
        self, logT: ArrayLike, logRho: ArrayLike, logz: ArrayLike
    ) -> ArrayLike:
        """__call__ method acting as convenience wrapper for the evaluate
        method. Calculates the (electron) conductive opacity
        of the gaseous mixture.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            logz (ArrayLike): log10 of the ion charge number.

        Returns:
            ArrayLike: (electron) conductive opacity
        """
        return self.evaluate(logT, logRho, logz)

    def __get_potekhin_table(self) -> ArrayLike:
        """Loads the Potekhin conductivity tables and returns
        the data as 3-dimensional array.

        Returns:
            ArrayLike: conductivity tables
        """
        num_entries = self.num_logTs * self.num_logRhos
        num_vals = 4
        table = np.loadtxt(self.table_path, skiprows=1)
        logTs = table[0, 1:]
        logRhos = table[1 : self.num_logRhos + 1, 0]
        logzs = table[0 :: self.num_logRhos + 1, 0]
        cond_data = np.zeros((self.num_logzs, num_entries, num_vals))
        i_start = 0
        for i in range(self.num_logzs):
            i_end = i_start + self.num_logRhos + 1
            part = table[i_start:i_end]
            part = part[1:]
            for j in range(self.num_logTs):
                isotherm = part[:, j + 1]
                j_start = j * self.num_logRhos
                j_end = j_start + self.num_logRhos
                cond_data[i, j_start:j_end, self.i_logT] = logTs[j]
                cond_data[i, j_start:j_end, self.i_logRho] = logRhos
                cond_data[i, j_start:j_end, self.i_logz] = np.log10(logzs[i])
                cond_data[i, j_start:j_end, self.i_logK] = isotherm
            i_start = i_end
        return cond_data

    def __get_rbs(self, iz: int, kx: int = 3, ky: int = 3) -> RectBivariateSpline:
        """Builds the RectBivariateSpline object for a
        given ion charge number.

        Args:
            iz (int): index of the ion charge number.
            kx (int, optional): degrees of the bivariate spline in logT.
                Defaults to 3.
            ky (int, optional): degrees of the bivariate spline in logRho.
                Defaults to 3.

        Returns:
            RectBivariateSpline: fitted RectBivariateSpline

        """
        logT = self.table[iz, :, self.i_logT]
        logRho = self.table[iz, :, self.i_logRho]
        X = np.unique(logT)
        Y = np.unique(logRho)
        Z = self.table[iz, :, self.i_logK]
        Z = np.reshape(Z, (X.size, Y.size))
        return RectBivariateSpline(X, Y, Z, kx=kx, ky=ky)

    def __get_rgi(self):
        """Builds the RegularGridInterpolator object for the
        conductivity in terms of (x, y, z) = (logT, logRho, logz).

        Returns:
            RegularGridInterpolator: fitted RegularGridInterpolator
        """
        x = self.logTs
        y = self.logRhos
        z = self.logzs
        data = np.zeros((self.num_logTs, self.num_logRhos, self.num_logzs))
        for i in range(self.num_logzs):
            data[:, :, i] = self.rbs[i](x, y, grid=True)
        rgi = RegularGridInterpolator(
            (x, y, z),
            data,
            method="pchip",
            bounds_error=False,
            fill_value=np.nan,
        )
        return rgi

    def evaluate_rbs(self, logT: float, logRho: float, logz: float) -> float:
        """Evaluates the RectBivariateSpline object for a given
        (logT, logRho, logz) and returns the (electron) conductive opacity.
        This only works with float inputs.

        Args:
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            logz (float): log10 of the ion charge number.

        Returns:
            float: conductive opacity.
        """
        if logz > np.max(self.logzs):
            i = self.num_logzs - 2
        else:
            i = (np.abs(self.logzs - logz)).argmin()
        j = i + 1
        logz0 = self.logzs[i]
        logz1 = self.logzs[j]

        rbs0 = self.rbs[i]
        rbs1 = self.rbs[j]
        logK0 = rbs0(logT, logRho)
        logK1 = rbs1(logT, logRho)
        logK = logK0 + (logz - logz0) * (logK1 - logK0) / (logz1 - logz0)
        # logK: log10 of the thermal conductivity
        # K: 10**logK (cgs units)
        # conductive opacity: kappa = 16 * sigma_b * T^3 / (3 * rho * K)
        logkap = 3 * logT - logRho - logK + np.log10(16 * sigma_b / 3)
        kap = np.power(10, logkap)
        return kap

    def evaluate_conductivity(self, logT: float, logRho: float, logz: float) -> float:
        """Evaluates the RegularGridInterpolator object for a given
        (logT, logRho, logz) and returns the (electron) conductivity.

        Args:
            logT (float): log10 of the temperature.
            logRho (float): log10 of the density.
            logz (float): log10 of the ion charge number.

        Returns:
            float: conductivity
        """
        if np.isscalar(logz):
            logz = logz * np.ones_like(logT)
        pts = (logT, logRho, logz)

        # logK: log10 of the thermal conductivity
        # conductive opacity: 16 * sigma_b * T^3 / (3 * rho * K)
        logK = self.rgi(pts)
        return logK

    def evaluate(
        self, logT: ArrayLike, logRho: ArrayLike, logz: ArrayLike
    ) -> ArrayLike:
        """Evaluates the RegularGridInterpolator object for a given
        (logT, logRho, logz) and returns the (electron) conductive opacity.

        Args:
            logT (ArrayLike): log10 of the temperature.
            logRho (ArrayLike): log10 of the density.
            logz (ArrayLike): log10 of the ion charge number.

        Returns:
            ArrayLike: conductive opacity.
        """
        logK = self.evaluate_conductivity(logT, logRho, logz)
        logkap = 3 * logT - logRho - logK + np.log10(16 * sigma_b / 3)
        kap = np.power(10, logkap)
        return kap
