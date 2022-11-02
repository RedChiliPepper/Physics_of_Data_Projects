import sys
sys.path.append("/home/eleonora/Scrivania/Magistrale/ComplexSystems/8286236/code/code/")
from GetRank import *

import scipy.io
import pandas as pd
# import numpy as np
# import csv
# import matplotlib.pyplot as plt

# plt.rcParams['text.usetex'] = True

np.seterr('raise')

mat = scipy.io.loadmat('turns_mobbing/group_05.mat')
your_data = mat['tracks']
tracks = pd.DataFrame(your_data)
# tracks.to_csv('turns_transit/group_03.csv') #your_data.csv final data in csv file

tracks = tracks.to_numpy()

Rank, Delay = GetRank(tracks)
print(Rank)
print(Delay)

Density, Order = GetDensityOrder(tracks)

print(Density)
print(Order)

Rank = np.sort(Rank)  # get information transfer distance
InfoDis = (Rank / Density) ** (1. / 3)
Delay = np.sort(Delay)

InfoDis_fit = InfoDis[(Delay < 1.8) & (Delay > 0.05)]
Delay_fit = Delay[(Delay < 1.8) & (Delay > 0.05)]

coef = np.polyfit(Delay_fit, InfoDis_fit, 1)
# coef = np.polyfit(Delay, InfoDis, 1)
poly1d_fn = np.poly1d(coef)

print(coef)

plt.scatter(Delay, InfoDis, color='coral')
plt.plot(Delay, poly1d_fn(Delay), '--k', color='black', linewidth=2)
plt.grid()

plt.xlabel('Time t (s)')
plt.ylabel('Distance x (m)')
plt.title('Distance travelled (velocity corr function), group_05')
plt.show()

RankAcc, DelayAcc = GetRankAcc(tracks)
print(RankAcc)
print(DelayAcc)
# GetRanktrial(tracks)

DensityAcc, OrderAcc = GetDensityOrder(tracks)

print(DensityAcc)
print(OrderAcc)

RankAcc = np.sort(RankAcc)  # get information transfer distance
InfoDisAcc = (RankAcc / DensityAcc) ** (1. / 3)
DelayAcc = np.sort(DelayAcc)

InfoDisAcc_fit = InfoDisAcc[(DelayAcc < 1.5) & (DelayAcc > 0.4)]
DelayAcc_fit = DelayAcc[(DelayAcc < 1.5) & (DelayAcc > 0.4)]

coefAcc = np.polyfit(DelayAcc_fit, InfoDisAcc_fit, 1)
# coefAcc = np.polyfit(DelayAcc, InfoDisAcc, 1)
poly1d_fn = np.poly1d(coefAcc)

print(coefAcc)

plt.scatter(DelayAcc, InfoDisAcc, color='coral')
plt.plot(DelayAcc, poly1d_fn(DelayAcc), '--k', color='black', linewidth=2)
plt.grid()

plt.xlabel('Time t (s)')
plt.ylabel('Distance x (m)')
plt.title('Distance travelled (acceleration corr function), group_05')
plt.show()
