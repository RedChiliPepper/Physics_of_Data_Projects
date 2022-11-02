# Generated with SMOP  0.41
# from smop.libsmop import *
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
#plt.rcParams['text.usetex'] = True


# GetRank.m

def GetRanktrial(tracks):
    BirdIDs = np.unique(tracks[:, 0])
    id = tracks[tracks[:, 0] == BirdIDs[0]]
    u1 = id[:, [5, 6, 7]]
    id1 = tracks[tracks[:, 0] == BirdIDs[-1]]
    u2 = id1[:, [5, 6, 7]]
    print('trial: ', u1, '\n', u2)
    dt = tracks[1, 4] - tracks[0, 4]
    cor_ut, cor_u, delay_temp = Cor(u1, u2, dt)
    print('results', cor_ut, cor_u, delay_temp)


# @function
def GetRank(tracks):
    # ss = pd.DataFrame(tracks[-300:,:]#from mpl_toolkits.mplot3d import Axes3D)
    # ss.to_csv('300.csv') #your_data.csv final data in csv file
    # print(tracks[-300:,:])
    BirdIDs = np.unique(tracks[:, 0])
    Score = np.zeros(len(BirdIDs))
    dt = tracks[1, 4] - tracks[0, 4]
    DelayM = np.zeros((len(BirdIDs), len(BirdIDs)))

    for i in range(len(BirdIDs)):
        ### for each bird, correlated with the rest of the group, delay matrix
        id = tracks[tracks[:, 0] == BirdIDs[i]]
        u1 = id[:, [5, 6, 7]]
        # print('BirdIDs[i]', BirdIDs[i])
        # if(i > (len(BirdIDs) - 10)): print('u1', u1[0:20,:])

        for j in range(len(BirdIDs)):
            if j == i:
                DelayM[i, j] = 0
                continue
            id = tracks[tracks[:, 0] == BirdIDs[j]]
            u2 = id[:, [5, 6, 7]]
            cor_ut, cor_u, delay_temp = None, None, None
            cor_ut, cor_u, delay_temp = Cor(u1, u2, dt)
            DelayM[i, j] = delay_temp
            if DelayM[i, j] < 0:
                Score[i] = Score[i] - 1
            elif DelayM[i, j] > 0:
                Score[i] = Score[i] + 1

    # print(DelayM[0:14, 0:14])

    ########### only use birds turn earlier
    # I = Score.argsort()[::-1]
    I = Score.argsort()
    Rank = I.argsort()
    # Score_new = Score[I]
    Delay = np.zeros(len(I))
    # print('rank', Rank + 1)
    #### calculate the delay time
    Delay[Rank.astype(int) == 0] = 0

    for i in range(1, len(I), 1):

        DelaySum = 0
        # idx = np.where(Rank.astype(int) == i)
        idx = Rank.astype(int).tolist().index(i)

        for j in range(0, i, 1):
            # idy = np.where(Rank.astype(int) == j)
            idy = Rank.astype(int).tolist().index(j)
            # print(idx, idy)
            Delay_temp = DelayM[idx, idy]
            DelaySum = DelaySum + Delay_temp + Delay[Rank.astype(int) == j]

        Delay[Rank.astype(int) == i] = DelaySum / i
    #### redo for some cases Delay < 0
    id = np.argwhere(Delay < 0)
    Delay[id] = 0
    Rank[id] = 0

    tij = []  # np.zeros(DelayM.shape[0])
    tikkj = []  # np.zeros(DelayM.shape[1])

    for i in range(DelayM.shape[0]):
        for j in range(DelayM.shape[1]):
            tij.append(DelayM[i, j])
            tikkj_temp = np.zeros(DelayM.shape[0])
            for k in range(DelayM.shape[0]):
                tikkj_temp[k] = DelayM[i, k] + DelayM[k, j]

            tikkj.append(np.mean(tikkj_temp))

    plt.scatter(tij, tikkj, color='teal')
    plt.plot(np.linspace(-2.5, 2.5, 500), np.linspace(-2.5, 2.5, 500), color='black')
    plt.xlabel('$t_{ij}$ (s)')
    plt.ylabel('$t_{ik} + t_{kj}$ (s)')
    plt.title('Correlation function using velocity')
    plt.show()

    return Rank, Delay


def Cor(u1, u2, dt):
    #### get delay time based on correlation in velocity,
    #### delay time is u1 relative to u2, i.e., delay>0, u1 turn later,
    # # #     #### normalize the u1 and u2
    # # #     U1=sum(u1.*u1,2).^0.5;
    # # #     U2=sum(u2.*u2,2).^0.5;
    # # #     u1=u1./U1;
    # # #     u2=u2./U2;
    ########### start the calculation
    Mid = int(np.round(np.shape(u1)[0] / 2))
    Slab = 30
    u2_t = u2[Mid - Slab - 1: Mid + Slab, :]
    n = np.shape(u2_t)[0]
    cor_ut = np.arange(Slab + 1 - Mid, np.shape(u1)[0] - Mid - Slab, 1)
    cor_u = np.zeros(len(cor_ut))
    for k in range(len(cor_ut)):
        u1_t = u1[Mid + cor_ut[k] - Slab - 1: Mid + cor_ut[k] + Slab, :]
        cor_u[k]=np.sum(np.sum(np.multiply(u1_t,u2_t)))
        cor_un = (np.sum(np.sum(u1_t ** 2)) ** 0.5) * (np.sum(np.sum(u2_t ** 2)) ** 0.5)
        """
        cor_u[k] = (np.sum(np.sum(np.multiply(u1_t, u2_t))) / n - (
                np.sum(np.sum(u1_t ** 2, axis=1) ** 0.5) * np.sum(np.sum(u2_t ** 2, axis=1) ** 0.5)) / (n ** 2))
        sigma1_t = (np.sum(np.sum(u1_t ** 2)) / n - (
                np.sum(np.sum(u1_t ** 2, axis=1) ** 0.5) * np.sum(np.sum(u1_t ** 2, axis=1) ** 0.5)) / (
                            n ** 2)) ** 0.5
        sigma2_t = (np.sum(np.sum(u2_t ** 2)) / n - (
                np.sum(np.sum(u2_t ** 2, axis=1) ** 0.5) * np.sum(np.sum(u2_t ** 2, axis=1) ** 0.5)) / (
                            n ** 2)) ** 0.5
        cor_un = sigma1_t * sigma2_t
        """
        cor_u[k] = cor_u[k] / cor_un

    # print(cor_u)
    cor_ut = cor_ut * dt
    # plt.scatter(cor_ut, cor_u)
    # plt.show()
    delay = cor_ut[np.argmax(cor_u)]
    # print('delay: ', delay)
    return cor_ut, cor_u, delay


def Cortrial(u1, u2, dt):
    Mid = int(np.round(np.shape(u1)[0] / 2))
    Slab = 2
    u2_t = u2[Mid - Slab: Mid + Slab, :]  # takes only central rows, from -Slab to +Slab wrt to the center
    print(u2_t)
    cor_ut = np.arange(Slab - Mid, np.shape(u1)[0] - Mid - Slab,
                       1)  # sequence, not sure why it was constructed this way
    print(cor_ut)
    cor_u = np.zeros(len(cor_ut))
    for k in range(len(cor_ut)):
        u1_t = u1[Mid + cor_ut[k] - Slab: Mid + cor_ut[k] + Slab,
               :]  # same number of rows and columns as u2_t, just translated
        print(cor_ut[k])
        print(u1_t)
        print(np.multiply(u1_t, u2_t))
        print(sum(np.inner(u1_t, u2_t)))
        print(sum(sum(np.multiply(u1_t, u2_t))))
        cor_u[k] = sum(sum(np.multiply(u1_t, u2_t)))  # sum by columns and rows
        cor_un = (sum(sum(u1_t ** 2)) ** 0.5) * (sum(sum(u2_t ** 2)) ** 0.5)
        cor_u[k] = cor_u[k] / cor_un
        if k == 0: print(cor_u[k])

    # print(cor_u)
    cor_ut = cor_ut * dt
    delay = cor_ut[np.argmax(cor_u)]
    print('delay: ', delay)
    return cor_ut, cor_u, delay


def GetDensityOrder(tracks):
    BirdIDs = np.unique(tracks[:, 0])
    T = np.unique(tracks[:, 4])
    Info_size = len(BirdIDs)
    p = np.zeros(len(T))
    D = np.zeros(len(T))

    for i in range(len(T)):
        id = tracks[tracks[:, 4] == T[i]]
        xyz = id[:, [1, 2, 3]]
        u = id[:, [5, 6, 7]]
        p[i] = 1. / np.shape(u)[0] * np.sum((np.sum(u / np.sum(u ** 2, 1)[:, None] ** 0.5, 0)) ** 2) ** 0.5
        D_temp = np.zeros(np.shape(xyz)[0])
        for j in range(np.shape(xyz)[0]):
            Dist_rank = np.sum((xyz - xyz[j, :]) ** 2, 1) ** 0.5
            D_temp[j] = np.max(Dist_rank)

        D[i] = Info_size / np.mean(D_temp) ** 3 * 6 / np.pi

    plt.scatter(T, p, color='teal')
    plt.title('Polarization')
    plt.xlabel('Time (s)')
    plt.ylabel('Polarization $\phi$')
    plt.show()
    Order = np.mean(p)
    Density = np.mean(D)

    return Density, Order


def GetRankAcc(tracks):
    # ss = pd.DataFrame(tracks[-300:,:])
    # ss.to_csv('300.csv') #your_data.csv final data in csv file
    # print(tracks[-300:,:])
    BirdIDs = np.unique(tracks[:, 0])
    Score = np.zeros(len(BirdIDs))
    dt = tracks[1, 4] - tracks[0, 4]
    DelayM = np.zeros((len(BirdIDs), len(BirdIDs)))

    for i in range(len(BirdIDs)):
        ### for each bird, correlated with the rest of the group, delay matrix
        id = tracks[tracks[:, 0] == BirdIDs[i]]
        a1 = id[:, [8, 9, 10]]
        # print('BirdIDs[i]', BirdIDs[i])
        # if(i > (len(BirdIDs) - 10)): print('u1', u1[0:20,:])

        for j in range(len(BirdIDs)):
            if j == i:
                DelayM[i, j] = 0
                continue
            id = tracks[tracks[:, 0] == BirdIDs[j]]
            a2 = id[:, [8, 9, 10]]
            cor_ut, cor_u, delay_temp = None, None, None
            cor_ut, cor_u, delay_temp = CorAcc(a1, a2, dt)
            DelayM[i, j] = delay_temp
            if DelayM[i, j] < 0:
                Score[i] = Score[i] - 1
            elif DelayM[i, j] > 0:
                Score[i] = Score[i] + 1

    # print(DelayM[0:14, 0:14])

    ########### only use birds turn earlier
    # I = Score.argsort()[::-1]
    I = Score.argsort()
    Rank = I.argsort()
    print(Score[I])
    Delay = np.zeros(len(I))
    # print('rank', Rank + 1)
    #### calculate the delay time
    Delay[Rank.astype(int) == 0] = 0

    for i in range(1, len(I), 1):

        DelaySum = 0
        # idx = np.where(Rank.astype(int) == i)
        idx = Rank.astype(int).tolist().index(i)

        for j in range(0, i, 1):
            # idy = np.where(Rank.astype(int) == j)
            idy = Rank.astype(int).tolist().index(j)
            # print(idx, idy)
            Delay_temp = DelayM[idx, idy]
            DelaySum = DelaySum + Delay_temp + Delay[Rank.astype(int) == j]

        Delay[Rank.astype(int) == i] = DelaySum / i
    #### redo for some cases Delay < 0
    id = np.argwhere(Delay < 0)
    Delay[id] = 0
    Rank[id] = 0

    tij = []  # np.zeros(DelayM.shape[0])
    tikkj = []  # np.zeros(DelayM.shape[1])

    for i in range(DelayM.shape[0]):
        for j in range(DelayM.shape[1]):
            tij.append(DelayM[i, j])
            tikkj_temp = np.zeros(DelayM.shape[0])
            for k in range(DelayM.shape[0]):
                tikkj_temp[k] = DelayM[i, k] + DelayM[k, j]

            tikkj.append(np.mean(tikkj_temp))

    plt.scatter(tij, tikkj, color='teal')
    plt.plot(np.linspace(-2.5, 2.5, 500), np.linspace(-2.5, 2.5, 500), color='black')
    plt.xlabel('$t_{ij}$ (s)')
    plt.ylabel('$t_{ik} + t_{kj}$ (s)')
    plt.title('Correlation function using acceleration')
    plt.show()

    return Rank, Delay


def CorAcc(a1, a2, dt):
    #### get delay time based on correlation in velocity,
    #### delay time is u1 relative to u2, i.e., delay>0, u1 turn later,
    # # #     #### normalize the u1 and u2
    # # #     U1=sum(u1.*u1,2).^0.5;
    # # #     U2=sum(u2.*u2,2).^0.5;
    # # #     u1=u1./U1;
    # # #     u2=u2./U2;
    ########### start the calculation
    Mid = int(np.round(np.shape(a1)[0] / 2))
    Slab = 30
    a2_t = a2[Mid - Slab - 1: Mid + Slab, :]
    n = np.shape(a2_t)[0]
    cor_at = np.arange(Slab + 1 - Mid, np.shape(a1)[0] - Mid - Slab, 1)
    cor_a = np.zeros(len(cor_at))
    for k in range(len(cor_at)):
        a1_t = a1[Mid + cor_at[k] - Slab - 1: Mid + cor_at[k] + Slab, :]
        cor_a[k]=(np.sum(np.sum(np.multiply(a1_t,a2_t)))/n - (np.sum(np.sum(a1_t ** 2, axis = 1) ** 0.5 )*np.sum(np.sum(a2_t ** 2, axis = 1) ** 0.5))/ (n**2))
        sigma1_t = (np.sum(np.sum(a1_t ** 2))/ n - (np.sum(np.sum(a1_t ** 2, axis = 1) ** 0.5 )*np.sum(np.sum(a1_t ** 2, axis = 1) ** 0.5))/ (n**2)) ** 0.5
        sigma2_t = (np.sum(np.sum(a2_t ** 2))/ n - (np.sum(np.sum(a2_t ** 2, axis = 1) ** 0.5 )*np.sum(np.sum(a2_t ** 2, axis = 1) ** 0.5))/ (n**2)) ** 0.5
        cor_an = sigma1_t*sigma2_t
        #cor_a[k] = np.sum(np.sum(np.multiply(a1_t, a2_t)))
        #cor_an = (np.sum(np.sum(a1_t ** 2)) ** 0.5) * (np.sum(np.sum(a2_t ** 2)) ** 0.5)
        cor_a[k] = cor_a[k] / cor_an
        # if k == 0: print(cor_a[k])

    # print(cor_a)
    cor_at = cor_at * dt
    # plt.scatter(cor_at, cor_a)
    # plt.show()
    delay = cor_at[np.argmax(cor_a)]
    # print('delay: ', delay)
    return cor_at, cor_a, delay





def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk


def update_lines(num, walks, lines):
    N = len(walks[0]) #25
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[max(num-N-1,0):num, :2].T)
        line.set_3d_properties(walk[max(num-N-1,0):num, 2])
    return lines


def PlotTracks(A):
    #Fixing random state for reproducibility
    np.random.seed(19680801)
    #A = tracks.to_numpy()
    BirdIDs = np.unique(A[:, 0])

    all_trajectories = []
    for i in range(len(BirdIDs)):
    ### for each bird, correlated with the rest of the group, delay matrix
        id = A[A[:, 0] == BirdIDs[i]]
        trajectory_i = id[:, [1, 2, 3]]
        all_trajectories.append(trajectory_i)

    # Data: 40 random walks as (num_steps, 3) arrays
    num_steps = len(all_trajectories[0])
    walks = all_trajectories

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = plt.axes(projection='3d') # fig.add_subplot(projection="3d")

    # Create lines initially without data
    lines = [ax.plot([], [], [], alpha=0.8, linewidth=0.75)[0] for _ in walks]

    # Setting the axes properties
    ax.set(xlim3d=(min(A[:,1]), max(A[:,1])), xlabel='X')
    ax.set(ylim3d=(min(A[:,2]), max(A[:,2])), ylabel='Y')
    ax.set(zlim3d=(min(A[:,3]), max(A[:,3])), zlabel='Z')

    # Creating the Animation object
    ani = animation.FuncAnimation(fig, update_lines, num_steps, fargs=(walks, lines), interval=1)


    #writergif = animation.FFMpegWriter(fps=60)
    #ani.save('flocks_m05.mp4', writer = writergif)

    plt.show()
