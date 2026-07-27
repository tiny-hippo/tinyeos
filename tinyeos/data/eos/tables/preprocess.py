import os

import numpy as np
from tinyeos.tableloader import TableLoader

T = TableLoader(which_hhe="cms")

# Convert effective hydrogen table
# from (logT, logP) to (logT, logRho)
# T.invert_xeff_pt_table(
#     kind="linear",
#     extrapolate=True,
#     smooth_table=False,
#     num_smoothing_rounds=2,
#     store_table=True,
# )

# Convert QEOS tables from (logT, logRho) to (logT, logP)
# for element in ["h2o", "sio2", "fe", "co"]:
#     T.invert_z_table(
#         which_variables="pt",
#         which_heavy=element,
#         kind="pchip",
#         extrapolate=True,
#         smooth_table=True,
#         num_smoothing_rounds=1,
#         store_table=True,
#     )

# Convert heavy-element tables from (logT, logP) to (logT, logRho)
elements = ["h2o", "fe", "mg2sio4"]
prefixes = ["aqua_revised", "paleos", "aneos"]
for i, element in enumerate(elements):
    T.invert_z_table(
        which_variables="dt",
        which_heavy=f"{prefixes[i]}_{element}",
        fname=f"{prefixes[i]}_dt_{element}.data",
        kind="pchip",
        extrapolate=True,
        smooth_table=False,
        store_table=True,
    )

# # Convert SESAME table from (logT, logP) to (logT, logRho)
# T.invert_z_table(
#     which_variables="dt",
#     which_heavy="sesame_h2o",
#     kind="pchip",
#     fname="sesame_dt_h2o.data",
#     extrapolate=True,
#     smooth_table=False,
#     store_table=True,
# )

# # Create a 50-50 qeos h2o-sio2 mixture pt table
# T.mix_heavy_elements(
#     which_Z1="h2o",
#     Z1=0.5,
#     which_Z2="sio2",
#     Z2=0.5,
#     which_Z3="fe",
#     Z3=0,
#     store_table=True,
# )

# # create smoothed dt tables with QEOS
# num_smoothing_rounds = 2
# for element in ["h2o", "sio2", "fe", "co", "mixture"]:
#     T = TableLoader(which_heavy=element)
#     table = T.z_dt_table
#     smoothed_table = T.smooth_z_table(table, num_smoothing_rounds=num_smoothing_rounds)
#     if element == "mixture":
#         element = "h2o_50_sio2_50_fe_00"
#     fname = f"qeos_smoothed_dt_{element}.data"
#     dst = os.path.join(T.tables_path, fname)
#     np.savetxt(dst, smoothed_table, fmt="%.8e", header=T.z_dt_header)

# # create smoothed pt tables with QEOS
# for element in ["h2o", "sio2", "fe", "co", "mixture"]:
#     T = TableLoader(which_heavy=element)
#     table = T.z_pt_table
#     smoothed_table = T.smooth_z_table(table, num_smoothing_rounds=num_smoothing_rounds)
#     if element == "mixture":
#         element = "h2o_50_sio2_50_fe_00"
#     else:
#         fname = f"qeos_smoothed_pt_{element}.data"
#     dst = os.path.join(T.tables_path, fname)
#     np.savetxt(dst, smoothed_table, fmt="%.8e", header=T.z_dt_header)
