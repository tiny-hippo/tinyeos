""" Example program that attempts to infer the
heavy-element profile for a given (logT, logRho, LogP).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from tinyeos.tinypteos import TinyPT
from tinyeos.support import get_X
from tinyeos.definitions import i_logRho

mpl.rcParams["lines.linewidth"] = 2.5

# load the profile
planet_profile = np.loadtxt("planet_profile.data")
# exclude regions colder than 100K
planet_profile = planet_profile[np.where(planet_profile[:, 2] > 100)]
logP = np.log10(planet_profile[:, 0]) + 10  # GPa to barye
R = planet_profile[:, 1]  # in Earth radii
logT = np.log10(planet_profile[:, 2])
logRho = np.log10(planet_profile[:, 3])


def func_PT(Z, logT, logP, logRho) -> float:
    """Support function for the heavy-element fraction
    root find using the pressure-temperature equation of state.

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


def get_Z(which_heavy) -> np.ndarray:
    """ Attempt to find the root in Z with the
    brentq algorithm in the bracket [a=0, b=1]. This
    assumes proto-solar hydrogen-helium ratio.

    Args:
        which_heavy (str): Which heavy-element equation
        of state to use.

    Returns:
        z_root_pt (np.ndarray): Heavy-element profile
    """

    # initiate the equation of state
    global tpt  # make sure the other function can access this variable
    tpt = TinyPT(which_heavy, which_hhe="cms")

    z_root_pt = np.zeros_like(logP)
    for i, lP in enumerate(logP):
        lT = logT[i]
        lRho = logRho[i]
        try:
            sol = brentq(func_PT, a=0, b=1, args=(lT, lP, lRho))
            z_root_pt[i] = sol
        except ValueError:
            # f(a) and f(b) have the same signs
            z_root_pt[i] = np.nan
    return z_root_pt


# get the heavy-element profiles
z_water = get_Z("water")
z_rock = get_Z("rock")
z_mix = get_Z("mixture")

# plot the heavy-element profile
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(R, z_water, label=r"$Z_{H_20}$")
ax.plot(R, z_rock, label=r"$Z_{SiO_2}$")
ax.plot(R, z_mix, label=r"$Z_{mix}$")
ax.set_xlim(left=0, right=4)
ax.set_ylim(bottom=0, top=1)
ax.set_xlabel(r"Radius [R$_\oplus$]")
ax.set_ylabel(r"Heavy-element fraction")
fig.legend(fancybox=True, framealpha=0.1, facecolor="black")
fig.tight_layout()
plt.show()
