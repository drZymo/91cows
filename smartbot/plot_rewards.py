import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import numpy as np

if len(sys.argv) > 1:
    RewardLogFile = Path(sys.argv[1])
else:
    RewardLogFile = Path('rewards.log')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def read():
    with open(RewardLogFile, 'r') as file:
        lines = file.readlines()
    rew = [float(line) for line in lines]
    return rew

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def animate(i):
    ax1.clear()
    #ax1.set_ylim(-22,22)
    rew = read()

    ax1.plot(rew, 'o', alpha=0.2)

    N = min(250, int(len(rew)/2))
    m = running_mean(rew, N)
    halfdiff = int((len(rew)-len(m))/2)
    ax1.plot(range(halfdiff,halfdiff+len(m)),m,'-', alpha=1, linewidth=3, color='black')

ani = animation.FuncAnimation(fig, animate, interval=10000)

plt.show()

