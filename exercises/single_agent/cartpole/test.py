#!/usr/bin/env python
import gymnasium as gym
import numpy as np
from DQNs import DQN
import pickle
from algorithms.utils.enviroment_tester import enviroment_tester_nn

def main():
    env =gym.make("CartPole-v1", render_mode = "human")

    name ="policy_net_dict.pkl"
    # q_dict = torch.load(name)
    with open(name, 'rb') as f:
        q_dict = pickle.load(f)
    nS = len(env.reset()[0])
    nA = env.unwrapped.action_space.n
    q = DQN(nS, nA)
    q.load_state_dict(q_dict)

    seed = np.random.randint(100, 10000)
    failing_state, reward_tot = enviroment_tester_nn(env, q, seed = seed, time_sleep=0.01).test()
    env.close()


if __name__== '__main__':
    main()