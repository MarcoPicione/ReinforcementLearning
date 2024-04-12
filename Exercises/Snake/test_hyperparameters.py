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
global num

rows = 27
cols = 27
num = 1000

def main():
    folder = "saved_q_tables/"
    files = os.listdir(folder)

    q_tables = []
    for file in files:
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            q_tables.append(np.load(path))

    means = np.zeros(len(q_tables))
    for idx, q in enumerate(q_tables):
        for i in range(num):
            seed = np.random.randint(100, 10000)
            env = gym.make("snake-v0", rows=rows, cols=cols, render_mode=None)
            env = TimeLimit(env, max_episode_steps=1000)
            state = env.reset(seed=seed)[0]
            terminated = False
            truncated = False
            while (not terminated and not truncated):
                action = np.argmax(q[state])
                new_state, reward, terminated, truncated, info = env.step(action)
                state = new_state
                score = env.unwrapped.snake.score
            env.close()
            means[idx] += score

    for i in range(len(means)):
        print(files[i], " : ", means[i] / num)


    

if __name__== '__main__':
    main()