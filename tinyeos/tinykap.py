import os
import pickle
import numpy as np
from typing import Tuple
from pathlib import Path
from scipy.interpolate import interp1d, RegularGridInterpolator
from tinyeos.support import get_FeH_from_Z, get_Z_from_FeH


class TinyFreedmanKap:
    def __init__(self, build_interpolants: bool = False) -> None:
        self.FeHs = np.array([0, 0.5, 0.7, 1.0, 1.5, 1.7])
        self.num_FeHs = self.FeHs.size
        self.cache_path = Path(__file__).parent / "data/kap"

        if build_interpolants:
            for FeH in self.FeHs:
                Z = get_Z_from_FeH(FeH)
                self.build_freedman_kap_interp(Z)

        self.rgis = []
        for FeH in self.FeHs:
            Z = get_Z_from_FeH(FeH)
            rgi = self.load_freedman_kap_interp(Z)
            self.rgis.append(rgi)

    def __call__(self, T: float, P: float, Z: float) -> float:
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

    @staticmethod
    def get_freedman_kap_fname(Z: float) -> str:
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

    def get_freedman_kap_tables(self, Z: float) -> Tuple:
        i_T = 0
        i_P = 1
        i_kap = 3
        fname = self.get_freedman_kap_fname(Z)
        src = os.path.join(self.cache_path, fname)
        with open(src, "r") as file:
            data = np.loadtxt(file, skiprows=38, dtype=np.float64)
        data = np.transpose(data)
        return (data[i_T, :], data[i_P, :], data[i_kap, :])

    def get_freedman_kap_from_tables(
        self, T: float, P: float, Z: float, num_Ts: int = 4, kind: str = "linear"
    ) -> float:
        Ts, Ps, kappas = self.get_freedman_kap_tables(Z)
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

    def get_freedman_kap_interp(self, Z: float) -> RegularGridInterpolator:
        Ts, Ps, _ = self.get_freedman_kap_tables(Z)
        unique_Ts = np.unique(Ts)
        unique_Ps = np.unique(Ps)

        num_Ts = unique_Ts.size
        num_Ps = unique_Ps.size
        kap = np.zeros((num_Ts, num_Ps))
        for i, T in enumerate(unique_Ts):
            for j, P in enumerate(unique_Ps):
                kap[i, j] = self.get_freedman_kap_from_tables(
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

    def build_freedman_kap_interp(self, Z: float) -> None:
        fname = self.get_freedman_kap_fname(Z)
        fname = fname.replace("txt", "pkl")
        dst = os.path.join(self.cache_path, fname)
        rgi = self.get_freedman_kap_interp(Z)
        with open(dst, "wb") as file:
            pickle.dump(rgi, file)

    def load_freedman_kap_interp(self, Z: float) -> RegularGridInterpolator:
        fname = self.get_freedman_kap_fname(Z)
        fname = fname.replace("txt", "pkl")
        src = os.path.join(self.cache_path, fname)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"missing interpolant cache {src}")
        with open(src, "rb") as file:
            rgi = pickle.load(file)
        return rgi


if __name__ == "__main__":
    tfk = TinyFreedmanKap(build_interpolants=False)
    T = 200
    P = 1e6
    Z = 0.012
    kap = tfk(T, P, Z)
    print(kap)
