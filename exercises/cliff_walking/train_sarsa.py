#!/usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
import gymnasium as gym
from gym.wrappers import TimeLimit
from algorithms.bootstrapping_methods.n_step_sarsa import n_step_sarsa
from algorithms.temporal_difference_methods.sarsa import sarsa
from copy import deepcopy
    
def main():
    num_episodes = 3000
    params ={'eps' : 1,
             'learning_rate' : 0.1,
             'discount_factor' : 1,
             'eps_decay_rate' : 1 / num_episodes
            }
    
    is_slippery = False
    env = gym.make("CliffWalking-v0")
    env = TimeLimit(env, max_episode_steps=1000)
    trainer = n_step_sarsa(10, env, num_episodes, params)
    trainer.run()
    env.close()

if __name__== '__main__':
    main()