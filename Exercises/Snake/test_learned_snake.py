#!/usr/bin/env python
import SnakeEnv
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from gym.wrappers import TimeLimit

global rows
global cols
global tot_episodes
rows = 22
cols = 22
tot_episodes = 1000000

def main():
    name = "q_tables/q_table_400000.npy"
    name = "learned_q_"+str(tot_episodes)+".npy"
    q = np.load(name)
    print(q)

    input()
    seed = 123
    env = gym.make("snake-v0", rows=rows, cols=cols, render_mode="human")
    state = env.reset(seed=seed)[0]
    terminated = False
    reward_tot = 0
    while not terminated:
        action = np.argmax(q[tuple(state)])
        print("Action ", action)
        new_state, reward, terminated, _, info = env.step(action)
        reward_tot += reward
        print(reward_tot)
        print("State ",state)
        print("food ", env.unwrapped.snake.food_pos)
        print("snake ", env.unwrapped.snake.snake_head_pos_prev)
        time.sleep(0.05)
        state = new_state



    env.close()


    

if __name__== '__main__':
    main()