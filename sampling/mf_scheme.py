"""
This file contains methods that (optimally & algorithmically) balance cost and expected improvement, based upon increasing the fidelity.
Most simple model is just increasing the resolution by e.g. 2 for new simulations.
Full model would incomporate the statistics of the current response surface (e.g. expected improvement), 
with the (additional) expected improvement based upon the accuracy model and balance it with the cost model
NOTE fixed cost per iteation? incorporate parallel sampling
TODO ego-ei voor incraesing mf? nieuw ontwikkelen, of blijft hetzelfde? literatuur?
"""


