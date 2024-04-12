#!/usr/bin/env python
import numpy as np
import os
from matplotlib import pyplot as plt
import gymnasium as gym
from gym.wrappers import TimeLimit
import time
import pickle

def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="human")
    env.unwrapped.render_mode = None
    epsilon = 0.9
    min_epsilon = 0.1
    max_epsilon = 1.0
    total_episodes = 100000
    decay_rate = 1 / total_episodes

    max_steps = 100

    lr_rate = 0.11
    gamma = 0.96

    Q = np.zeros((env.observation_space.n, env.action_space.n))
        
    def choose_action(state):
        action=0
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        return action

    def learn(state, state2, reward, action, action2):
        predict = Q[state, action]
        target = reward + gamma * Q[state2, action2]
        Q[state, action] = Q[state, action] + lr_rate * (target - predict)

    # Start
    rewards=0

    for episode in range(total_episodes):
        print(episode)
        t = 0
        state = env.reset()
        state = int(state[0])
        action = int(choose_action(state))
        
        while t < max_steps:
            #env.render()

            state2, reward, done,done2, info = env.step(action)

            action2 = choose_action(state2)

            learn(state, state2, reward, action, action2)

            state = int(state2)
            action = int(action2)

            t += 1
            rewards+=1

            if done or done2:
                break
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode) 
    # os.system('clear')
            # time.sleep(0.1)

        
    print ("Score over time: ", rewards/total_episodes)
    print(Q)

    with open("frozenLake_qTable_sarsa.pkl", 'wb') as f:
        pickle.dump(Q, f)

    env.unwrapped.render_mode="human"
    state = env.reset()[0]
    terminated = False
    while not terminated:
        action = np.argmax(Q[state,:])
        print("Action ", action)
        new_state, reward, terminated, _, info = env.step(action)
        print("New state ",new_state)
        state = new_state
    env.close()
    

if __name__== '__main__':
    main()