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
tot_episodes = 10000000


def run(episodes, use_nn, render = False):
    env = gym.make("snake-v0", rows=rows, cols=cols, render_mode="human" if render else None)
    env = TimeLimit(env, max_episode_steps=10000)

    q = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, env.action_space.n))


    lr = 0.9
    df = 0.9
    eps = 1
    eps_decay_rate = 1 / episodes / 100
    rng = np.random.default_rng()

    # plot stuff
    rewards = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        while (not terminated and not truncated):
            state_idx = tuple(state)
            if rng.random() < eps:
                action = env.action_space.sample()  # agent policy that uses the observation and info
                # print("random ", action)
            else:
                action = np.argmax(q[state_idx])
                # print("not random ", action, "state ", state)
            
            new_state, reward, terminated, truncated, info = env.step(action)
            # if(reward > 1): print("Gotcha")

            new_state_idx = tuple(new_state)
            action_state_idx = tuple(state) + (action,)

            q[action_state_idx] = q[action_state_idx] + lr * (reward + df * np.max(q[new_state_idx]) - q[action_state_idx])
            state = new_state
            
            if terminated or truncated:
                observation, info = env.reset()
            
        eps = max(eps - eps_decay_rate, 0)
        if(eps==0):
            lr = 0.0001

        # plot stuff
        if(reward == 1):
            rewards[i] = 1

        if (i % (tot_episodes / 100) == 0): 
            print("Training ", i / tot_episodes * 100, " %", end='\r')
            name = "./q_tables/q_table_"+str(i)+".npy"
            np.save(name, q)

    env.close()

    # plot stuff
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards[max(0, t-1000):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('snake'+str(rows)+'x'+str(cols)+'.png')
    name = "learned_q_"+str(tot_episodes)+".npy"
    np.save(name, q)

def main():
    train = True
    name = "learned_q_"+str(tot_episodes)+".npy"
    if not os.path.isfile(name) or train:
        run(tot_episodes, render = False)
    
    q = np.load(name)
    print(q)
    env = gym.make("snake-v0", rows=rows, cols=cols, render_mode="human")
    state = env.reset()[0]
    terminated = False
    while not terminated:
        action = np.argmax(q[tuple(state)])
        print("Action ", action)
        new_state, reward, terminated, _, info = env.step(action)
        print(reward)
        time.sleep(0.1)
        print("New state ",new_state)
        state = new_state

    env.close()


    

if __name__== '__main__':
    main()