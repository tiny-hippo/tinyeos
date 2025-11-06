## tinyeos
tinyeos is a python package to calculate the equation of state of a mixture
of hydrogen, helium and a heavy element.

### Installation
Download or clone this repository, navigate into the directory and execute
```
pip install .
```

Note that the package requires numpy, scipy, fortranformat.

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
