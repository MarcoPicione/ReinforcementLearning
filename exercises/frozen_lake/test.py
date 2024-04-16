#!/usr/bin/env python

import numpy as np
import gymnasium as gym
from algorithms.utils.enviroment_tester import enviroment_tester

def main():
    q = np.load("learned_q_FrozenLake-v1_10000_episodes_dyna_q.npy")
    print(q)
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human")
    enviroment_tester(env, q).test()
    env.close()

if __name__== '__main__':
    main()