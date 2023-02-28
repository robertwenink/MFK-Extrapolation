" This file is used to make plots of reconstruction made. These are based on coordinates per given reconstruction line. "
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

timeFile = 0
timeIndex = 0

base_path = os.path.dirname(os.path.abspath(__file__))

dirFilePLIC = base_path + "/Excel/Reconstruction/" + "ReconstPLIC_" + str(timeFile) + "s.csv"
dirFiledx = base_path + "/dx.txt"
dirFiledy = base_path + "/dy.txt"

" Plot the structure in space "
DataPLIC = np.genfromtxt(dirFilePLIC, delimiter=',')
dx, dy = np.genfromtxt(dirFiledx, delimiter=',')[2:-2], np.genfromtxt(dirFiledy, delimiter=',')[2:-2]
N, M = len(dx), len(dy)

plt.figure()
if len(DataPLIC) != 0:
    pntsPLIC = np.linspace(0, int(DataPLIC.shape[1] - 2), int(DataPLIC.shape[1] / 2), endpoint=True)
    for xx in pntsPLIC:
        x, y = np.sum(dx[:int(DataPLIC[0, int(xx + 1.0)]) - 2]), np.sum(dy[:int(DataPLIC[0, int(xx)]) - 2])
        DataPLIC[1:, int(xx)], DataPLIC[1:, int(xx + 1.0)] = DataPLIC[1:, int(xx)] + x, DataPLIC[1:, int(xx + 1.0)] + y
        if xx == pntsPLIC[-1]:
            plt.plot(DataPLIC[1:, int(xx)], np.sum(dy) - DataPLIC[1:, int(xx) + 1], 'k', label="PLIC reconstruction")
        else:
            plt.plot(DataPLIC[1:, int(xx)], np.sum(dy) - DataPLIC[1:, int(xx) + 1], 'k')
    
"Grid"
for xx in range(0, len(dx)+1):
    plt.plot([np.sum(dx[:xx]), np.sum(dx[:xx])], [0, np.sum(dy)], 'k', linewidth=0.2)
for xx in range(0, len(dy)+1):
    plt.plot([0, np.sum(dx)], [np.sum(dy[xx:]), np.sum(dy[xx:])], 'k', linewidth=0.2)

plt.xlabel("X")
plt.ylabel('Y')
plt.legend(loc = 'upper left')
plt.axis('equal')
plt.show()
