#!/usr/bin/env python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

def run(episodes, is_slippery, render = False):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=is_slippery, render_mode="human" if render else None)
    q = np.zeros((env.observation_space.n, env.action_space.n))

    lr = 0.9
    df = 0.9
    eps = 1
    eps_decay_rate = 1 / episodes
    rng = np.random.default_rng()

    # plot stuff
    rewards = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        while (not terminated and not truncated):
            if rng.random() < eps:
                action = env.action_space.sample()  # agent policy that uses the observation and info
            else:
                action = np.argmax(q[state, :])
            
            new_state, reward, terminated, truncated, info = env.step(action)

            # if(reward == 1):
            #     print("Gotcha")

            q[state, action] = q[state, action] + lr * (reward + df * np.max(q[new_state,:]) - q[state,action])
            state = new_state
            
            if terminated or truncated:
                observation, info = env.reset()
            
        eps = max(eps - eps_decay_rate, 0)
        if(eps==0):
            lr = 0.0001

        # plot stuff
        if(reward == 1):
            rewards[i] = 1

    env.close()

    # plot stuff
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    name = "learned_q_" + str(is_slippery) + ".npy"
    np.save(name, q)

def main():
    train = True
    is_slippery = False
    name = "learned_q_" + str(is_slippery) + ".npy"
    if not os.path.isfile(name) or train:
        run(50000, False, render = False)
    
    q = np.load(name)
    print(q)
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=is_slippery, render_mode="human")
    state = env.reset()[0]
    terminated = False
    while not terminated:
        action = np.argmax(q[state, :])
        print("Action ", action)
        new_state, reward, terminated, _, info = env.step(action)
        print("New state ",new_state)
        state = new_state

    env.close()





if __name__== '__main__':
    main()