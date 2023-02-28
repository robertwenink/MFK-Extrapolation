import os
import matplotlib.pyplot as plt
from core.sampling.solvers.solver import get_solver
output_path = os.path.join(os.getcwd(),"belangrijke EVA runs gebruikt in rapport","06975_09443 refinement 3.0 voor extra x refinement")
solver = get_solver(name = "EVA")
solver.inspect_results(output_path, only_ax2 = True, alternative_sup_title = "Before", legend = True)
plt.show()