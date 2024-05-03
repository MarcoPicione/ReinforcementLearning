#!/usr/bin/env python
import gymnasium as gym
from DQNs import DQN
import torch

from algorithms.approximation_methods.deep_RL import deep_RL
   
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
    
    env =gym.make("CartPole-v1")
    trainer = deep_RL(parameters, DQN, env, 1000)
    trainer.run()

if __name__ == "__main__":
    main()
     


