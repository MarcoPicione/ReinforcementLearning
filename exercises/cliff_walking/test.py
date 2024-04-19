#!/usr/bin/env python

import numpy as np
import gymnasium as gym
from algorithms.utils.enviroment_tester import enviroment_tester

def main():
    # q = np.load("learned_q_CliffWalking-v0_10000_episodes_.npy")
    q = np.load("learned_q_CliffWalking-v0_3000_episodes_10_step_sarsa.npy")
    # q = np.load("learned_q_CliffWalking-v0_1000_episodes_mc_eps_greedy.npy")
    print(q)
    env = gym.make("CliffWalking-v0", render_mode="human")
    enviroment_tester(env, q).test()
    env.close()

if __name__== '__main__':
    main()