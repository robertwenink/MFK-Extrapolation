" This file is used to make plots of reconstruction made. These are based on coordinates per given reconstruction line. "
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

timeFile = 0
timeIndex = 0
xspeed = 1
yspeed = 1
x_orig = [0.1 + timeFile * xspeed, 0.9 + timeFile * xspeed]
y_orig = [0.1 + timeFile * yspeed, 0.9 + timeFile * yspeed]

base_path = os.path.dirname(os.path.abspath(__file__))
files = glob.glob(base_path + "/Excel/Reconstruction/ReconstBLIC_*.csv")
files.sort(key=os.path.getmtime)

dirFilePLIC = base_path + "/Excel/Reconstruction/" + "ReconstPLIC_" + str(timeFile) + "s.csv"
dirFileBLIC = base_path + "/Excel/Reconstruction/" + "ReconstBLIC_" + str(timeFile) + "s.csv"
dirFilePLIC0 = base_path + "/Excel/Reconstruction/" + "ReconstPLIC_0s.csv"
dirFileBLIC0 = base_path + "/Excel/Reconstruction/" + "ReconstBLIC_0s.csv"
dirFiledx = base_path + "/dx.txt"
dirFiledy = base_path + "/dy.txt"

dirFileBLIC = files[timeIndex]

" Plot the structure in space "
DataPLIC = np.genfromtxt(dirFilePLIC, delimiter=',')
DataBLIC = np.genfromtxt(dirFileBLIC, delimiter=',')
DataPLIC0 = np.genfromtxt(dirFilePLIC0, delimiter=',')
DataBLIC0 = np.genfromtxt(dirFileBLIC0, delimiter=',')
dx, dy = np.genfromtxt(dirFiledx, delimiter=',')[2:-2], np.genfromtxt(dirFiledy, delimiter=',')[2:-2]
N, M = len(dx), len(dy)
pntsBLIC = np.linspace(0, int(DataBLIC.shape[1] - 2), int(DataBLIC.shape[1] / 2))
pntsBLIC0 = np.linspace(0, int(DataBLIC0.shape[1] - 2), int(DataBLIC0.shape[1] / 2))

plt.figure()
if len(DataPLIC) != 0:
    pntsPLIC = np.linspace(0, int(DataPLIC.shape[1] - 2), int(DataPLIC.shape[1] / 2))
    for xx in pntsPLIC:
        x, y = np.sum(dx[:int(DataPLIC[0, int(xx + 1.0)]) - 2]), np.sum(dy[:int(DataPLIC[0, int(xx)]) - 2])
        DataPLIC[1:, int(xx)], DataPLIC[1:, int(xx + 1.0)] = DataPLIC[1:, int(xx)] + x, DataPLIC[1:, int(xx + 1.0)] + y
        if xx == pntsPLIC[-1]:
            plt.plot(DataPLIC[1:, int(xx)], np.sum(dy) - DataPLIC[1:, int(xx)+1], 'k', label="PLIC")
        else:
            plt.plot(DataPLIC[1:, int(xx)], np.sum(dy) - DataPLIC[1:, int(xx) + 1], 'k')

if np.sum(DataBLIC[0, :2]) != 0:
    for xx in pntsBLIC:
        x, y = np.sum(dx[:int(DataBLIC[0, int(xx + 1.0)]) - 2]), np.sum(dy[:int(DataBLIC[0, int(xx)]) - 2])
        DataBLIC[1:, int(xx)], DataBLIC[1:, int(xx + 1.0)] = DataBLIC[1:, int(xx)] + x, DataBLIC[1:, int(xx + 1.0)] + y
        BLICp = DataBLIC[3, int(xx):int(xx+2.0)].copy()
        DataBLIC[3, int(xx):int(xx+2.0)] = DataBLIC[2, int(xx):int(xx+2.0)]
        DataBLIC[2, int(xx):int(xx+2.0)] = BLICp
        if xx == pntsBLIC[-1]:
            plt.plot(DataBLIC[1:, int(xx)], np.sum(dy) - DataBLIC[1:, int(xx)+1], 'r', label="BLIC")
        else:
            plt.plot(DataBLIC[1:, int(xx)], np.sum(dy) - DataBLIC[1:, int(xx) + 1], 'r')
# " Initial "
# if len(DataPLIC0) != 0:
#     pntsPLIC0 = np.linspace(0, int(DataPLIC0.shape[1] - 2), int(DataPLIC0.shape[1] / 2))
#     for xx in pntsPLIC0:
#         x, y = np.sum(dx[:int(DataPLIC0[0, int(xx + 1.0)]) - 2]), np.sum(dy[:int(DataPLIC0[0, int(xx)]) - 2])
#         DataPLIC0[1:, int(xx)], DataPLIC0[1:, int(xx + 1.0)] = DataPLIC0[1:, int(xx)] + x + timeFile * xspeed, DataPLIC0[1:, int(xx + 1.0)] + y - timeFile * yspeed
#         if xx == pntsPLIC0[-1]:
#             plt.plot(DataPLIC0[1:, int(xx)], np.sum(dy) - DataPLIC0[1:, int(xx)+1], 'k', label="PLIC0")
#         else:
#             plt.plot(DataPLIC0[1:, int(xx)], np.sum(dy) - DataPLIC0[1:, int(xx) + 1], 'k')
#
# if np.sum(DataBLIC0[0, :2]) != 0:
#     for xx in pntsBLIC0:
#         x, y = np.sum(dx[:int(DataBLIC0[0, int(xx + 1.0)]) - 2]), np.sum(dy[:int(DataBLIC0[0, int(xx)]) - 2])
#         DataBLIC0[1:, int(xx)], DataBLIC0[1:, int(xx + 1.0)] = DataBLIC0[1:, int(xx)] + x + timeFile * xspeed, DataBLIC0[1:, int(xx + 1.0)] + y - timeFile * yspeed
#         BLICp = DataBLIC0[3, int(xx):int(xx+2.0)].copy()
#         DataBLIC0[3, int(xx):int(xx+2.0)] = DataBLIC0[2, int(xx):int(xx+2.0)]
#         DataBLIC0[2, int(xx):int(xx+2.0)] = BLICp
#         if xx == pntsBLIC0[-1]:
#             plt.plot(DataBLIC0[1:, int(xx)], np.sum(dy) - DataBLIC0[1:, int(xx)+1], 'k', label="BLIC0")
#         else:
#             plt.plot(DataBLIC0[1:, int(xx)], np.sum(dy) - DataBLIC0[1:, int(xx) + 1], 'k')

" Original "
plt.plot([x_orig[0], x_orig[1], x_orig[1], x_orig[0], x_orig[0]], [y_orig[0], y_orig[0], y_orig[1], y_orig[1], y_orig[0]], 'y', label="Original")

"Grid"
for xx in range(0, len(dx)):
    plt.plot([np.sum(dx[:xx]), np.sum(dx[:xx])], [0, np.sum(dy)], 'k', linewidth=0.2)
for xx in range(0, len(dy)):
    plt.plot([0, np.sum(dx)], [np.sum(dy[:xx]), np.sum(dy[:xx])], 'k', linewidth=0.2)

plt.legend()

plt.show()
