""" Simple example program showing how to call the equation of state. """
import numpy as np
from tinyeos.tinypteos import TinyPT
from tinyeos.definitions import i_logRho

# Create the equation of state object.
Tpt = TinyPT(which_heavy="co", which_hhe="cms")

# Define input temperature, pressure and composition.
logT = 3.00
logP = 10
X = 0.60
Z = 0.20

# Evaluate the equation of state at the given
# pressure, temperature and composition. The result is a (18,) array
# containting the equation of state output (see definitions module for
# available quantities). """
eos_result = Tpt.evaluate(logT, logP, X, Z)

# Get only pressure from the result and print it.
logRho = eos_result[i_logRho]
print(f"logRho = {logRho:.2f}")
