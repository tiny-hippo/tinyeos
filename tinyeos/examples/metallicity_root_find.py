""" Example program that attempts to infer the
heavy-element profile for a given (logT, logRho, LogP).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tinyeos.tinypteos import TinyPT
from tinyeos.support import get_X
from tinyeos.definitions import i_logRho


def get_Z(which_heavy) -> np.ndarray:
    """Attempt to find the root in Z with the
    brentq algorithm in the interval [min_Z, max_Z]. This
    assumes proto-solar hydrogen-helium ratio.

    Args:
        which_heavy (str): Which heavy-element equation
            of state to use.

    Returns:
        z_root_pt (np.ndarray): Heavy-element profile
    """
    # upper and lower bracket for brentq
    min_Z = 0
    max_Z = 1

    # initiate the equation of state
    tpt = TinyPT(which_heavy, which_hhe="cms")

    def func_PT(Z, logT, logP, logRho) -> float:
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

    z_root_pt = np.zeros_like(logP)
    for i in range(logP.size):
        try:
            sol = brentq(
                func_PT,
                a=min_Z,
                b=max_Z,
                args=(logT[i], logP[i], logRho[i])
            )
            z_root_pt[i] = sol
        except ValueError:
            # func_PT(min_Z) and func_PT(max_Z) have the same signs
            z_root_pt[i] = np.nan
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
z_water = get_Z("water")
z_rock = get_Z("rock")
z_mix = get_Z("mixture")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(R, z_water, color="tab:purple", label=r"$Z_{H_20}$")
ax.plot(R, z_rock, color="tab:grey", label=r"$Z_{SiO_2}$")
ax.plot(R, z_mix, color="tab:pink", label=r"$Z_{mix}$")
ax.set_xlim(left=0, right=4)
ax.set_ylim(bottom=0, top=1)
ax.set_xlabel(r"Radius [R$_\oplus$]")
ax.set_ylabel(r"Heavy-element fraction")
fig.legend()
fig.tight_layout()
plt.show()
