#!/usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
import gymnasium as gym
from gym.wrappers import TimeLimit
from algorithms.sarsa import sarsa
from algorithms.n_step_sarsa import n_step_sarsa

def main():
    num_episodes = 100000
    params ={'eps' : 1,
             'learning_rate' : 0.001,
             'discount_factor' : 1,
             'eps_decay_rate' : 1 / num_episodes / 100
            }
    
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=is_slippery, render_mode="None")
    env = TimeLimit(env, max_episode_steps=100)
    trainer = n_step_sarsa(5, env, num_episodes, params)
    trainer.run()
    env.close()

if __name__== '__main__':
    main()