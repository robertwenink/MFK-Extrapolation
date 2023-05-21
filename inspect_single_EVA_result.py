import os

from matplotlib import pyplot as plt
from core.sampling.solvers.external import EVA

from utils.formatting_utils import correct_formatX


" Script to print the best point easily "
solver = EVA()

# LOW MASS
# Wedge_optimization_constant_mass_low\2.0\10000_05086
x = [0.8809, 0.5214]
refinement = 2.0
solver.base_path = os.path.normpath("./EVA_Results/Wedge_optimization_constant_mass_low")
solver.inspect_results(solver.get_output_path(x, refinement)[0])
plt.show()

# HIGH MASS
# Wedge_optimization_constant_mass_high\2.0\05676_0.0044
x = [0.5676, 0.0044]
refinement = 2.0
solver.base_path = os.path.normpath("./EVA_Results/Wedge_optimization_constant_mass_high")
solver.inspect_results(solver.get_output_path(x, refinement)[0])
plt.show()