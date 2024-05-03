#!/usr/bin/env python
import exercises.single_agent.snake.snake_env as snake_env
import gymnasium as gym
import numpy as np
import time
from algorithms.utils.enviroment_tester import enviroment_tester

def main():
    name ="learned_q_snake-v0_100000_episodes_sarsa.npy"
    #name ="learned_q_snake-v0_300000_episodes_2_step_sarsa.npy"

    q = np.load(name)
    seed = np.random.randint(100, 10000)
    env = gym.make("snake-v0", rows=27, cols=27, render_mode="human")
    failing_state, reward_tot = enviroment_tester(env, q, seed = seed, time_sleep=0.01).test()
    env.close()

    print("SCORE: ", env.unwrapped.snake.score)
    print("REWARD: ", reward_tot)
    print("SEED ", seed)
    print("FAILING STATE ", failing_state)


    

if __name__== '__main__':
    main()