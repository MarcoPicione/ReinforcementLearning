#!/usr/bin/env python
import exercises.snake.snake_env as snake_env
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from gym.wrappers import TimeLimit
from algorithms.bootstrapping_methods.n_step_sarsa import n_step_sarsa
from algorithms.temporal_difference_methods.qlearning import qlearning
from algorithms.temporal_difference_methods.sarsa import sarsa

def main():
    num_episodes = 100000
    env = gym.make("snake-v0", rows=12, cols=12, render_mode=None)
    env = TimeLimit(env, max_episode_steps=1000)
    params ={'eps' : 1,
             'learning_rate' : 0.1,
             'discount_factor' : 1,
             'eps_decay_rate' : 1 / num_episodes
            }
    
    trainer = n_step_sarsa(1, env, num_episodes, params)
    trainer.run()
    env.close()   

if __name__== '__main__':
    main()