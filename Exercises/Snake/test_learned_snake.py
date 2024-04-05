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
    name = "saved_q_tables/q_no_dist.npy"
    q = np.load(name)
    # print(q)

    # input()
    seed = np.random.randint(100, 10000)
    seed = 2264
    env = gym.make("snake-v0", rows=rows, cols=cols, render_mode="human")
    env = TimeLimit(env, max_episode_steps=10)
    state = env.reset(seed=seed)[0]
    terminated = False
    truncated = False
    reward_tot = 0
    while (not terminated and not truncated):
        action = np.argmax(q[tuple(state)])
        new_state, reward, terminated, truncated, info = env.step(action)
        reward_tot += reward
        time.sleep(0.01)
        state = new_state
        score = env.unwrapped.snake.score
    env.close()

    print("SCORE: ", score)
    print("SEED ", seed)


    

if __name__== '__main__':
    main()