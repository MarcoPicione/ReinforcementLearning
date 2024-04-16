#!/usr/bin/env python

import gymnasium as gym
from gym.wrappers import TimeLimit
from algorithms.dp_methods.policy_iteration import policy_iteration
from algorithms.utils.enviroment_tester import enviroment_tester
import numpy as np

def main():   
    is_slippery = False
#     env = gym.make('FrozenLake-v1', desc=[
#     "SFFHFFFF",
#     "FFFFFFFF",
#     "FFFHFFFF",
#     "FFFHFHFF",
#     "FFFHFFFF",
#     "FHHFFHHF",
#     "HHFFHFHH",
#     "FFFFFFFG",
# ], map_name="8x8", is_slippery=is_slippery, render_mode=None)
    env = gym.make('CliffWalking-v0')
    # env = TimeLimit(env, max_episode_steps=100)
    trainer = policy_iteration(env, 0.99, 0.0001)
    v, policy = trainer.run()
    # print(np.reshape(policy, (4, 12)))
    print(len(policy))
    enviroment_tester(env, policy=policy).test()
    env.close()

if __name__== '__main__':
    main()