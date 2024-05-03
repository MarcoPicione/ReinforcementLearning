#!/usr/bin/env python

import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
from algorithms.utils.enviroment_tester import enviroment_tester

def main():
    # q = np.load("learned_q_CliffWalking-v0_10000_episodes_.npy")
    q = np.load("learned_q_CliffWalking-v0_3000_episodes_sarsa.npy")
    q = np.load("learned_q_CliffWalking-v0_3000_episodes_.npy")
    print(q)
    env = gym.make("CliffWalking-v0", render_mode="human")
    enviroment_tester(env, q).test()
    env.close()

    qlearning = np.load("cumulative_reward_CliffWalking-v0_3000_episodes_.npy")
    sarsa = np.load("cumulative_reward_CliffWalking-v0_3000_episodes_sarsa.npy")
    plt.figure()
    plt.plot(qlearning, label = "q_learning")
    plt.plot(sarsa, label = "sarsa")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative reward of the last 300 episodes")
    plt.legend()
    plt.show()


if __name__== '__main__':

    main()