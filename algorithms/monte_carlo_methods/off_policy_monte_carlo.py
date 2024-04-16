################
# DOESN T WORK #
################

import numpy as np
from tqdm import tqdm
from algorithms.utils.plots import plot_cumulative_reward

class off_policy_monte_carlo():
    def __init__(self, env, episodes, params, b = None):
        self.env = env
        self.episodes = episodes
        self.eps = params['eps']
        self.df = params['discount_factor']
        self.eps_decay_rate = params['eps_decay_rate']
        self.rng = np.random.default_rng()
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        self.q = np.zeros((self.nS, self.nA))
        self.pi = np.zeros(self.nS)
        self.c = np.zeros((self.nS, self.nA))
        self.b = np.ones((self.nS, self.nA)) / self.nA if b is None else b
        self.rewards = np.zeros(self.episodes)

    def pick_action(self, state):
        return np.random.choice(np.arange(self.nA), p=self.b[state])
    
    def run(self):
        for i in tqdm(range(self.episodes), desc="Training", unit="iter"):
            episode_reward = 0

            # b = np.random.randint(0, self.nA, self.nS)
            states_action_list = []
            rewards_list = []

            state = self.env.reset()[0]
            terminated = False
            truncated = False

            while (not terminated and not truncated):
                action = self.pick_action(state)
                action_state_reward_idx = (state, action)
                states_action_list.append(action_state_reward_idx)
                state, reward, terminated, truncated, info = self.env.step(action)
                rewards_list.append(reward)
                episode_reward += reward

            G = 0
            W = 1
                                       
            for t in range(len(states_action_list) - 1, -1, -1):
                G = self.df * G + rewards_list[t]
                p = states_action_list[t]
                self.c[p] += W
        
                self.q[p] += W / self.c[p] * (G - self.q[p])
                self.pi[p[0]] = np.argmax(self.q[p[0]])

                if p[1] is not self.pi[p[0]]: break
                W /= self.b[p]
  
            self.eps = max(self.eps - self.eps_decay_rate, 0)
            self.rewards[i] = episode_reward

        # plot stuff
        env_name = self.env.spec.id
        s = env_name + "_" + str(self.episodes) + "_episodes_off_policy_mc"
        path = "reward_" + s + ".png"
        plot_cumulative_reward(self.episodes, self.rewards, path, label = "off policy monte carlo")

        # save stuff
        name = "learned_q_" + s + ".npy"
        np.save(name, self.q)
        name = "learned_policy_" + s + ".npy"
        np.save(name, self.pi)