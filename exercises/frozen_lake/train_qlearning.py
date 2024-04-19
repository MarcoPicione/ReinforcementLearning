#!/usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
import gymnasium as gym
from gym.wrappers import TimeLimit
from algorithms.temporal_difference_methods import qlearning

def main():
    num_episodes = 10000
    params ={'eps' : 1,
             'learning_rate' : 0.1,
             'discount_factor' : 0.9,
             'eps_decay_rate' : 1 / num_episodes
            }
    
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=is_slippery, render_mode="None")
    env = TimeLimit(env, max_episode_steps=100)
    trainer = qlearning(env, num_episodes, params)
    trainer.run()
    env.close()

if __name__== '__main__':
    main()