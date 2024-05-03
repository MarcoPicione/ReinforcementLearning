#!/usr/bin/env python
import gymnasium as gym
from gym.wrappers import TimeLimit
from algorithms.monte_carlo_methods.off_policy_monte_carlo import off_policy_monte_carlo

def main():
    num_episodes = 100000
    params ={'eps' : 1,
             'discount_factor' : 1,
             'eps_decay_rate' : 1 / num_episodes
            }
    
    is_slippery = False
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery, render_mode="None")
    env = TimeLimit(env, max_episode_steps=100)
    trainer = off_policy_monte_carlo(env, num_episodes, params)
    trainer.run()
    env.close()

if __name__== '__main__':
    main()