import numpy as np
import matplotlib.pyplot as plt
from tinyeos.tinypteos import TinyPT
from tinyeos.support import get_X
from tinyeos.definitions import i_logRho

# load the profile
planet_profile = np.loadtxt("planet_profile.data")
# exclude regions colder than 100K
planet_profile = planet_profile[np.where(planet_profile[:, 2] > 100)]
logP = np.log10(planet_profile[:, 0]) + 10  # GPa to barye
R = planet_profile[:, 1]  # in Earth radii
logT = np.log10(planet_profile[:, 2])
logRho = np.log10(planet_profile[:, 3])

# remove duplicates
logP, indices = np.unique(logP, return_index=True)
logRho = logRho[indices]
logT = logT[indices]

# initiate and call the equation of state with
# some arbitrary composition
# note: X and Z can also be arrays
Z = 0.50
X = get_X(Z)
tpt = TinyPT()
logRho_eos = tpt.evaluate(logT, logP, X, Z)[i_logRho]

# plot the two density profiles
fig, ax = plt.subplots(1, 1)
ax.plot(logP, logRho, lw=2)
ax.plot(logP, logRho_eos, lw=2)
ax.set_xlabel("logP (Ba)")
ax.set_ylabel("logRho (g/cc)")
fig.tight_layout()
plt.show()

# the equation of state can also
# handle two-dimensional inputs
num_pts = 512
logT = np.linspace(2, 4, num_pts)
logP = np.linspace(6, 10, num_pts)
# create a 2d grid spanned by (logT, logP)
logT, logP = np.meshgrid(logT, logP)
logRho = tpt.evaluate(logT, logP, X, Z)[i_logRho]

# make a contour plot of the density
fig, ax = plt.subplots(1, 1)
cf = ax.contourf(logT, logP, logRho, levels=50)
fig.colorbar(cf, label="logRho (g/cc)")
ax.set_xlabel("logT (K)")
ax.set_ylabel("logP (Ba)")
fig.tight_layout()
plt.show()
