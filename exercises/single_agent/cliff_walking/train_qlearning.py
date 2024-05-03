#!/usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
import gymnasium as gym
from gym.wrappers import TimeLimit
from algorithms.temporal_difference_methods.qlearning import qlearning

def main():
    num_episodes = 3000
    params ={'eps' : 1,
             'learning_rate' : 0.1,
             'discount_factor' : 0.9,
             'eps_decay_rate' : 1 / num_episodes
            }
    
    is_slippery = False
    env = gym.make("CliffWalking-v0")
    env = TimeLimit(env, max_episode_steps=100)
    # env.unwrapped.render_mode = "human"
    trainer = qlearning(env, num_episodes, params, save_cumulative_reward = True)
    trainer.run()
    env.close()

if __name__== '__main__':
    main()