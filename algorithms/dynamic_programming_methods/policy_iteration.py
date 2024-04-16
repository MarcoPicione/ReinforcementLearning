#!/usr/bin/env python

import numpy as np
from algorithms.utils.plots import plot_reward
from algorithms.utils.enviroment_tester import enviroment_tester
import gymnasium as gym
import warnings
warnings.filterwarnings("error")

class policy_iteration:
    def __init__(self, env, df, theta, generate_plot = False):
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.env = env
        self.df = df
        self.theta = theta
        self.v = np.zeros(self.nS)
        self.policy = np.zeros(self.nS, self.nA)
        self.generate_plot = generate_plot

    def policy_evaluation(self):
        while True:
            delta = 0
            for s in range(self.nS):
                v_previous = self.v[s]

                sum = 0
                # print(self.policy)
                for a in range(self.nA):#, action_prob in enumerate(self.policy[s]):
                    for prob, new_state, reward, _ in self.env.unwrapped.P[s][a]:
                        sum += 1 / self.nA * prob * (reward + self.df * self.v[new_state])

                # sum = 0
                # for a, action_prob in enumerate(self.policy[s]):
                #     for prob, new_state, reward, _ in self.env.unwrapped.P[s][a]:
                #         try:
                #             sum = sum + action_prob * prob * (reward + self.df * self.v[new_state])
                #             # if sum >10000: print(s)
                #         except:
                #             print(sum, prob, reward, self.v[new_state])
                self.v[s] = sum
                delta = max(delta, np.abs(v_previous - self.v[s]))
            if delta < self.theta: break
        return self.v
    
    def run(self):
        policy_stable = False
        it = 0
        if self.generate_plot:
            rewards = {}
        while(not policy_stable):
            print("it ", it)
            self.policy_evaluation()
            policy_stable = True
            for s in range(self.nS):
                old_action = self.policy[s] #np.argmax(self.policy[s])
                possible_actions = np.zeros(self.nA)
                for a in range(self.nA):
                    sum = 0
                    for prob, new_state, reward, _ in self.env.unwrapped.P[s][a]:
                        sum += prob * (reward + self.df * self.v[new_state])
                    possible_actions[a] = sum
                    # print(possible_actions)
                    # input()
                self.policy[s] = np.argmax(possible_actions)
                if old_action != self.policy[s]: policy_stable = False
            if self.generate_plot:
                _, r = enviroment_tester(self.env, policy = self.policy).test()
                rewards[it] = r
            it += 1
            # if it>10000:
            #     print(self.policy)
            #     input()

        
        if self.generate_plot:
            env_name = self.env.spec.id
            s = env_name + "_" + str(it) + "_iterations_"
            path = "reward_" + s + ".png"
            plot_reward(self.episodes, rewards, path)
        return self.v, self.policy
    
def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode=None)
    p_it = policy_iteration(env, 0.9, 1e-20)
    nS = env.observation_space.n
    nA = env.action_space.n
    # p_it.policy = np.ones([nS, nA]) / nA
    v = p_it.policy_evaluation()
    print(v)

if __name__== '__main__':
    main()

