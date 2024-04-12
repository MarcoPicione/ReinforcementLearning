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
from algorithms.sarsa import sarsa
from algorithms.n_step_sarsa import n_step_sarsa

global rows
global cols
global tot_episodes
rows = 12
cols = 12
tot_episodes = 10000


def run(episodes, render = False):
    # empty folder
    folder = "./q_tables/"
    files = os.listdir(folder)
    for file in files:
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            os.remove(path)
            

    env = gym.make("snake-v0", rows=rows, cols=cols, render_mode="human" if render else None)
    env = TimeLimit(env, max_episode_steps=1000)
    # Construct tuple
    # t = []
    # for i in range(2 + env.unwrapped.snake.snake_body_max):
    #         t += [env.unwrapped.rows, env.unwrapped.cols]

    q = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, env.action_space.n))
    # print(q.shape)

    lr = 0.1
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
        episode_reward = 0
        while (not terminated and not truncated):
            state_idx = tuple(state)
            if rng.random() < eps:
                action = env.action_space.sample()  # agent policy that uses the observation and info
                # print("random ", action)
            else:
                action = np.argmax(q[state_idx])
                # print("not random ", action, "state ", state)
            
            new_state, reward, terminated, truncated, info = env.step(action)
            # print("typeee ", type(new_state))
            # if(reward > 1): print("Gotcha")
            episode_reward += reward

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
        rewards[i] = episode_reward

        if (i % (tot_episodes / 100) == 0): 
            print("Training ", i / tot_episodes * 100, " %", end='\r')
            name = "./q_tables/q_table_"+str(i)+".npy"
            np.save(name, q)

    env.close()

    # plot stuff
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards[max(0, int(t-tot_episodes/100)):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('snake'+str(rows)+'x'+str(cols)+'.png')
    name = "learned_q_"+str(tot_episodes)+".npy"
    np.save(name, q)
    print(rewards)

def main():
    num_episodes = 10000
    env = gym.make("snake-v0", rows=rows, cols=cols, render_mode=None)
    env = TimeLimit(env, max_episode_steps=1000)
    params ={'eps' : 1,
             'learning_rate' : 0.005,
             'discount_factor' : 0.95,
             'eps_decay_rate' : 1 / num_episodes / 100
            }
    
    trainer = n_step_sarsa(1, env, num_episodes, params)
    trainer.run()
    env.close()   

if __name__== '__main__':
    main()