## tinyeos
tinyeos is a python package to calculate the equation of state of a mixture
of hydrogen, helium and a heavy element.

### Installation
Installation is simple: download or clone this repository, navigate into the directory and execute
```
pip install .
```
which is the preferred way to install.
Note that the package requires numpy, scipy, fortranformat and scikit-learn.

### Basic usage
```python
import tinyeos

logT = 4  # log10 of the temperature in K
logP = 10 # log10 of the pressure in Ba
X = 0.7  # hydrogen mass-fraction
Z = 0.1  # heavy-element mass-fraction

tpt = tinyeos.TinyPT()
res = tpt.evaluate(logT, logP, X, Z)
```
See the examples folder for more.
