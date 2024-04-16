import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from algorithms.utils.plots import plot_mean_reward

class n_step_sarsa():
    def __init__(self, n, env, episodes, params):
        self.n = n
        self.env = env
        self.episodes = episodes
        self.max_steps = env.spec.max_episode_steps
        self.eps = params['eps']
        self.lr = params['learning_rate']
        self.df = params['discount_factor']
        self.eps_decay_rate = params['eps_decay_rate']
        self.rng = np.random.default_rng() 
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.pi = np.zeros((self.env.observation_space.n)).astype(np.int_)
        self.rewards_plot = np.zeros(self.episodes)      

    def eps_greedy_pick(self, state):
        if self.rng.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = self.pi[state]
        return action

    def run(self):
        for i in tqdm(range(self.episodes), desc="Training", unit="iter"):
            episode_reward = 0
            T = self.max_steps
            t = -1
            tau = 0
            rewards = {}
            states = {}
            actions = {}

            state = self.env.reset()[0]
            state_idx = tuple(state) if isinstance(state, np.ndarray) else (state, )
            action = self.eps_greedy_pick(state_idx)
            
            states[0] = state_idx
            actions[0] = action
           
            terminated = False
            truncated = False

            while tau < (T-1):
                t += 1
                if t < T:
                    new_state, reward, terminated, truncated, info = self.env.step(action)
                    new_state_idx = tuple(new_state) if isinstance(new_state, np.ndarray) else (new_state, )
                    rewards[(t + 1) % (self.n + 1)] = reward
                    states[(t + 1) % (self.n + 1)] = new_state_idx
                    episode_reward += reward

                    if terminated or truncated:
                        T = t + 1
                    else:
                        action = self.eps_greedy_pick(new_state)
                        actions[(t + 1) % (self.n + 1)] = action

                tau = t - self.n + 1
                if tau >= 0:
                    G = np.sum([self.df**(idx - tau - 1) * rewards[idx % (self.n + 1)] for idx in range(tau + 1, min(tau + self.n, T) + 1)])
                    if (tau + self.n) < T: 
                        G += self.df ** self.n * self.q[states[(tau + self.n) % (self.n + 1)], actions[(tau + self.n) % (self.n + 1)]]
                    self.q[states[tau % (self.n + 1)], actions[tau % (self.n + 1)]] += self.lr * (G - self.q[states[tau % (self.n + 1)], actions[tau % (self.n + 1)]])
                    self.pi[states[tau % (self.n + 1)]] = np.argmax(self.q[states[tau % (self.n + 1)]])
                
            self.eps = max(self.eps - self.eps_decay_rate, 0)
            self.rewards_plot[i] = episode_reward

        env_name = self.env.spec.id
        s = env_name + "_" + str(self.episodes) + "_episodes_" + str(self.n) + "_step_sarsa"
        path = "reward_" + s + ".png"
        plot_mean_reward(self.episodes, self.rewards_plot, path, str(self.n) + "_step")

        # save stuff
        name = "learned_q_" + s + ".npy"
        np.save(name, self.q)
        name = "learned_policy_" + s + ".npy"
        np.save(name, self.pi)