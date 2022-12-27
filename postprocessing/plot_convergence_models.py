###########################################################################
## Extra Plotting options                                               ###
###########################################################################

# def draw_convergence(solver, solver_type_text):
#     # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     fig = plt.figure()
#     ax1 = fig.add_subplot(221, projection="3d")
#     ax2 = fig.add_subplot(222)
#     ax3 = fig.add_subplot(212)
#     fig.suptitle("Convergence type: {}".format(solver_type_text))
#     x = np.arange(0, 1 + np.finfo(np.float32).eps, 0.05)

#     "show convergence profile"
#     L = np.arange(10)
#     y = []
#     for l in L:
#         y.append(solver_wrapper(solver(l), x))
#     X, L_mesh = np.meshgrid(x, L)
#     y = np.array(y)
#     ax1.plot_surface(X, L_mesh, y, alpha=0.9)
#     # ax1.plot(x, L, y)
#     ax1.set_ylabel("Fidelity level")
#     ax1.set_xlabel("x location")
#     ax1.set_title("Convergence profile")

#     # "show costs"
#     ax2.plot(L, sampling_costs(L))
#     ax2.set_xlabel("Fidelity level")
#     ax2.set_title("Sampling costs")

#     "show mf function"
#     for l in range(0, 6, 1):
#         y, _ = mf_forrester2008(x, l, solver)
#         ax3.plot(x, y, label="Level = {}".format(l))
#     ax3.plot(x, forrester2008(x), ".-", label="Fully converged")
#     ax3.set_xlabel("X: 1D search space")
#     ax3.set_title("Evaluation function responses per fidelity")

#     # # ax1.set_aspect(1.0 / ax1.get_data_ratio(), adjustable="box")
#     # ax2.set_aspect(1.0 / ax2.get_data_ratio(), adjustable="box")
#     # ax3.set_aspect(1.0 / ax3.get_data_ratio(), adjustable="box")

#     plt.legend()
    
