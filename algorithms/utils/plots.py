import numpy as np
from matplotlib import pyplot as plt

def plot_cumulative_reward(episodes, rewards, path, label, same_plot = False):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards[max(0, int(t - episodes / 100)) : (t + 1)])

    if not same_plot: plt.figure()
    plt.plot(sum_rewards, label = label)
    plt.ylabel("Cumulative reward of last "+ str(int(episodes / 100)) + " episodes")
    plt.xlabel("Episodes")
    plt.legend()
    plt.savefig(path)
    
    return sum_rewards

def plot_mean_reward(episodes, rewards, path, label, same_plot = False):
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards[max(0, int(t - episodes / 100)) : (t + 1)]) / (episodes / 100)

    if not same_plot: plt.figure()
    plt.plot(sum_rewards, label = label)
    plt.ylabel("Mean reward of last "+ str(int(episodes / 100)) + " episodes")
    plt.xlabel("Episodes")
    plt.legend()
    plt.savefig(path)