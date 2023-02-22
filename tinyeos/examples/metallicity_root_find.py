""" Example program that attempts to infer the
heavy-element profile for a given (logT, logRho, LogP).
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.optimize import brentq
from tinyeos.tinypteos import TinyPT
from tinyeos.support import get_X
from tinyeos.definitions import i_logRho

mpl.rcParams["lines.linewidth"] = 2.5
mpl.rcParams["font.size"] = 12


def get_z(which_heavy, logT, logP, logRho) -> NDArray:
    """Attempt to find the root in Z with the
    brentq algorithm in the interval [min_Z, max_Z] for a given
    logT, logP and logRho profile. This assumes
    proto-solar hydrogen-helium ratio.

    Args:
        which_heavy (str): Which heavy-element equation
            of state to use.
        logT (ArrayLike): Log10 of the temperature
        logP (ArrayLike): Log10 of  the pressure
        logRho (ArrayLike): Log10 of the density

    Returns:
        z_root_pt (np.ndarray): Heavy-element profile
    """
    # initiate the equation of state
    tpt = TinyPT(which_heavy, which_hhe="cms", include_hhe_interactions=False)

    def func_pt(Z, logT, logP, logRho) -> float:
        """Support function for the heavy-element fraction
        root finding.

        Args:
            Z (float): Heavy-element mass fraction
            logT (float): Log10 of the temperature
            logP (float): Log10 of  the pressure
            logRho (float): Log10 of the density

        Returns:
            (float): logRho residual
        """

        X = get_X(Z)
        eos_result = tpt.evaluate(logT, logP, X, Z)
        return logRho - eos_result[i_logRho]

    # upper and lower bracket for brentq
    min_Z = 0
    max_Z = 1
    z_root_pt = np.zeros_like(logP)
    for i in range(logP.size):
        # check that the signs are different at the brackets
        f_lower = func_pt(min_Z, logT[i], logP[i], logRho[i])
        f_upper = func_pt(max_Z, logT[i], logP[i], logRho[i])
        if np.sign(f_lower) == np.sign(f_upper):
            # no root within brackets
            sol = np.nan
        else:
            sol = brentq(
                func_pt,
                a=min_Z,
                b=max_Z,
                args=(logT[i], logP[i], logRho[i])
                )
        z_root_pt[i] = sol
    return z_root_pt


# load the profile and exclude regions colder than 100K
planet_profile = np.loadtxt("planet_profile.data")
planet_profile = planet_profile[np.where(planet_profile[:, 2] > 100)]
# keep only every fourth grid point for faster computation
planet_profile = planet_profile[::4, :]
logP = np.log10(planet_profile[:, 0]) + 10  # GPa to barye
R = planet_profile[:, 1]  # in Earth radii
logT = np.log10(planet_profile[:, 2])
logRho = np.log10(planet_profile[:, 3])

# get the heavy-element profiles and plot them
z_water = get_z("h2o", logT, logP, logRho)
z_rock = get_z("sio2", logT, logP, logRho)
z_mix = get_z("mixture", logT, logP, logRho)
z_co = get_z("co", logT, logP, logRho)
z_fe = get_z("fe", logT, logP, logRho)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(R, z_water, label=r"H$_2$0")
ax.plot(R, z_co, label=r"CO")
ax.plot(R, z_rock, label=r"SiO$_2$")
ax.plot(R, z_mix, label=r"50-50 H$_2$O-SiO$_2$")
ax.plot(R, z_fe, label=r"Fe")
ax.set_xlim(left=0, right=4)
ax.set_ylim(bottom=0, top=1)
ax.set_xlabel(r"Radius [R$_\oplus$]")
ax.set_ylabel(r"Heavy-element fraction")
ax.legend()
fig.tight_layout()
plt.show()
