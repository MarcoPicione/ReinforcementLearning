#!/usr/bin/env python

import numpy as np
import gymnasium as gym
from pettingzoo.butterfly import pistonball_v6

def main():
    env = pistonball_v6.env(n_pistons=100, time_penalty=-0.1, continuous=True,
                      random_drop=True, random_rotate=True, ball_mass=0.75, ball_friction=0.3,
                      ball_elasticity=1.5, max_cycles=125, render_mode="human")
    env.reset()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = env.action_space(agent).sample()

            env.step(action)
    env.close()

if __name__== '__main__':
    main()