"""Definitions for the equation of state boundaries and the
array indices for the output.
"""

# currently supported heavy elements
heavy_elements = ["h2o", "sesame_h2o", "aqua", "sio2", "fe", "co", "mixture"]
atomic_masses = {
    "h2o": 18.015,
    "sesame_h2o": 18.015,
    "aqua": 18.015,
    "sio2": 60.080,
    "fe": 55.845,
    "co": 28.010,
    "mixture": 0.5 * (18.015 + 60.080),
}
ionic_charges = {
    "h2o": 10,
    "sesame_h2o": 10,
    "aqua": 10,
    "sio2": 30,
    "fe": 26,
    "co": 14,
    "mixture": 0.5 * (10 + 30),
}

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
eps2 = 1e-32
tiny_val = 1e-16
tiny_logRho = -32

eos_num_vals = 19
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

eos_dict = {
    "logT": i_logT,
    "logRho": i_logRho,
    "logP": i_logP,
    "logS": i_logS,
    "logU": i_logU,
    "chiRho": i_chiRho,
    "chiT": i_chiT,
    "grad_ad": i_grad_ad,
    "c_p": i_cp,
    "c_v": i_cv,
    "gamma1": i_gamma1,
    "gamma3": i_gamma3,
    "dS_dT": i_dS_dT,
    "dS_dRho": i_dS_dRho,
    "dE_dRho": i_dE_dRho,
    "mu": i_mu,
    "eta": i_eta,
    "lfe": i_lfe,
    "c_sound": i_csound,
}

# num_derivs = 3
# i_dl_dlT = 0
# i_dl_dlRho = 1
# i_dl_dlP = 2
