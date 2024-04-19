#!/usr/bin/env python
import gymnasium as gym
from gym.wrappers import TimeLimit
from algorithms.tabular_methods.dyna_q import dyna_q

def main():
    num_episodes = 10000
    params ={'eps' : 1,
             'discount_factor' : 0.9,
             'learning_rate' : 0.1,
             'eps_decay_rate' : 1 / num_episodes
            }
    
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=is_slippery, render_mode="human")
    env = TimeLimit(env, max_episode_steps=100)
    trainer = dyna_q(env, num_episodes, params, 10)
    trainer.run()
    env.close()

if __name__== '__main__':
    main()