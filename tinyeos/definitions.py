""" Definitions for the equation of state boundaries and the
array indices for the output.
"""

# min/max temperatures are limited by SCvH and CMS
logT_max = 6.00
logT_min = 2.00
# min/max pressures and densities are limited by QEoS
logP_max = 15.00
logP_min = 1.00
logRho_max = 2.00
logRho_min = -8.00

# tolerances and tiny values
eps1 = 1e-6
eps2 = 1e-8
tiny_val = 1e-16

num_vals = 19
i_logT = 0
i_logRho = 1
i_logP = 2
i_logS = 3
i_logU = 4
i_chiRho = 5
i_chiT = 6
i_grad_ad = 7
i_cp = 8
i_cv = 9
i_gamma1 = 10
i_gamma3 = 11
i_dS_dT = 12
i_dS_dRho = 13
i_dE_dRho = 14
i_mu = 15
i_eta = 16
i_lfe = 17
i_csound = 18

# num_derivs = 3
# i_dl_dlT = 0
# i_dl_dlRho = 1
# i_dl_dlP = 2
