#!/usr/bin/env python

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
import torch
from DQNs import DQN
from algorithms.approximation_methods.deep_RL import deep_RL
import exercises.single_agent.snake.snake_env as snake_env

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parameters = {'batch_size' : 128,
                  'gamma' : 0.99,
                  'eps_start' : 0.9,
                  'eps_end' : 0.05,
                  'eps_decay' : 1000,
                  'tau' : 0.005,
                  'lr' : 1e-4,
                  'memory_size' : 10000
                  }
    
    env = gym.make("snake-v0", rows=12, cols=12, render_mode=None, nn_trained = True)
    trainer = deep_RL(parameters, DQN, env, 1000)
    trainer.run()

if __name__== '__main__':
    main()