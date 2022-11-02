import scipy.io
import pandas as pd
# import numpy as np
# import csv
# import matplotlib.pyplot as plt
import sys

sys.path.append("/home/eleonora/Scrivania/Magistrale/ComplexSystems/8286236/code/code/")
from GetRank import *
#plt.rcParams['text.usetex'] = True

np.seterr('raise')


for i in range(1, 9):
    # mat = scipy.io.loadmat('turns_mobbing/group_05.mat')
    mat = scipy.io.loadmat('turns_mobbing/group_%02d.mat' % i)
    # mat = scipy.io.loadmat('turns_transit/group_12.mat')
    your_data = mat['tracks']
    tracks = pd.DataFrame(your_data)
    # tracks.to_csv('turns_transit/group_03.csv') #your_data.csv final data in csv file

    tracks = tracks.to_numpy()
    # print('tracks 4')
    # print(tracks[:, 4])

    # PlotTracks(tracks)

    # T = np.unique(tracks[:, 4])
    # move coordinate to along flight direction
    # id = tracks[tracks[:, 4] == T[0]]
    """
    u = id[:, [5, 6, 7]]
    U = u.mean(axis=0)
    
    theta_temp = np.arctan(U[1] / U[0])
    R = np.matrix([[np.cos(theta_temp), - np.sin(theta_temp)], [np.sin(theta_temp), np.cos(theta_temp)]])
    print(R)
    print(np.asarray(R.T * (np.matrix([U[0], U[1]])).T).reshape(-1))
    
    
    UR = np.append(np.asarray(R.T * (np.matrix([U[0], U[1]])).T).reshape(-1), U[2])
    
    for j in range(len(tracks[:, 0])):
        if UR[0] > 0:
            tracks[j, [1, 2]] = tracks[j, [1, 2]] * R
            tracks[j, [5, 6]] = tracks[j, [5, 6]] * R
        elif UR[0] < 0:
            tracks[j, [1, 2]] = tracks[j, [1, 2]] * (-R)
            tracks[j, [5, 6]] = tracks[j, [5, 6]] * (-R)
    
    N = len(np.unique(tracks[:, 0]))
    Rank, Delay = GetRank(tracks)
    print(Rank)
    print(Delay)
    #GetRanktrial(tracks)
    
    
    Density, Order = GetDensityOrder(tracks)
    
    print(Density)
    print(Order)
    
    Rank = np.sort(Rank)  # get information transfer distance
    InfoDis = (Rank / Density) ** (1./3)
    
    Delay = np.sort(Delay)
    plt.scatter(Delay, InfoDis)
    plt.show()
    """

    # N = len(np.unique(tracks[:, 0]))

    Rank, Delay = GetRank(tracks)
    print(Rank)
    print(Delay)
    # GetRanktrial(tracks)

    Density, Order = GetDensityOrder(tracks)

    print(Density)
    print(Order)

    Rank = np.sort(Rank)  # get information transfer distance
    InfoDis = (Rank / Density) ** (1. / 3)

    Delay = np.sort(Delay)
    plt.scatter(Delay, InfoDis)
    plt.xlabel('Time t (s)')
    plt.ylabel('Distance x (m)')
    plt.title('Distance travelled (velocity corr function), group_%02d' % i)
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
    plt.scatter(DelayAcc, InfoDisAcc)
    plt.xlabel('Time t (s)')
    plt.ylabel('Distance x (m)')
    plt.title('Distance travelled (acceleration corr function), group_%02d' % i)
    plt.show()
