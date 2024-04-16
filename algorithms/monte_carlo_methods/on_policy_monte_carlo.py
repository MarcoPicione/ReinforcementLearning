import numpy as np
from tqdm import tqdm
from algorithms.utils.plots import plot_cumulative_reward

class on_policy_monte_carlo():
    def __init__(self, env, episodes, params):
        self.env = env
        self.episodes = episodes
        self.eps = params['eps']
        self.df = params['discount_factor']
        self.eps_decay_rate = params['eps_decay_rate']
        self.rng = np.random.default_rng()
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        self.q = np.zeros((self.nS, self.nA))
        self.pi = np.ones((self.nS, self.nA)).astype(np.int_) / self.nA
        self.returns = {s : {a : [] for a in range(self.nA)} for s in range(self.nS)}
        self.rewards = np.zeros(self.episodes)

    def eps_greedy_pick(self, state):
        if self.rng.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.pi[state])
        return action
    
    def run(self):
        for i in tqdm(range(self.episodes), desc="Training", unit="iter"):
            episode_reward = 0

            states_action_list = []
            rewards_list = []

            state = self.env.reset()[0]
            terminated = False
            truncated = False

            while (not terminated and not truncated):
                action = self.eps_greedy_pick(state)
                action_state_reward_idx = (state, action)
                states_action_list.append(action_state_reward_idx)
                state, reward, terminated, truncated, info = self.env.step(action)
                rewards_list.append(reward)
                episode_reward += reward

            G = 0
                                       
            for t in range(len(states_action_list) - 1, -1, -1):
                G = self.df * G + rewards_list[t]
                p = states_action_list[t]
                if p not in states_action_list[: t - 1]:
                    self.returns[p[0]][p[1]].append(G)
                    self.q[p] = np.mean(self.returns[p[0]][p[1]])
                    best_action = np.argmax(self.q[p[0]])

                    self.pi[p[0]] = [self.eps / self.nA for i in range(self.nA)]
                    self.pi[p[0]][best_action] += 1 - self.eps
  
            self.eps = max(self.eps - self.eps_decay_rate, 0)
            self.rewards[i] = episode_reward

        # plot stuff
        env_name = self.env.spec.id
        s = env_name + "_" + str(self.episodes) + "_episodes_on_policy_mc"
        path = "reward_" + s + ".png"
        plot_cumulative_reward(self.episodes, self.rewards, path, label = "on policy monte carlo")

        # save stuff
        name = "learned_q_" + s + ".npy"
        np.save(name, self.q)
        name = "learned_policy_" + s + ".npy"
        np.save(name, self.pi)