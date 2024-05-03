#!/usr/bin/env python
import gymnasium as gym
import numpy as np
from DQNs import DQN
import pickle
from algorithms.utils.enviroment_tester_nn import enviroment_tester_nn
import exercises.single_agent.snake.snake_env as snake_env

def main():
    env = gym.make("snake-v0", rows=27, cols=27, render_mode=None, nn_trained = True)

    name ="policy_net_dict.pkl"
    # q_dict = torch.load(name)
    with open(name, 'rb') as f:
        q_dict = pickle.load(f)
    nS = len(env.reset()[0])
    nA = env.unwrapped.action_space.n
    q = DQN(nS, nA)
    q.load_state_dict(q_dict)

    seed = np.random.randint(100, 10000)
    failing_state, reward_tot = enviroment_tester_nn(env, q, seed = seed, time_sleep=0.05).test()
    env.close()


if __name__== '__main__':
    main()