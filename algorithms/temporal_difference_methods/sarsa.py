import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithms.utils.plots import plot_cumulative_reward

class sarsa():
    def __init__(self, env, episodes, params, save_cumulative_reward = False):
        self.env = env
        self.episodes = episodes
        self.eps = params['eps']
        self.lr = params['learning_rate']
        self.df = params['discount_factor']
        self.eps_decay_rate = params['eps_decay_rate']
        self.save_cumulative_reward = save_cumulative_reward
        self.rng = np.random.default_rng() 
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.pi = np.zeros((self.env.observation_space.n)).astype(np.int_)
        self.rewards = np.zeros(self.episodes)

    def eps_greedy_pick(self, state):
        if self.rng.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = self.pi[state]
        return action

    def run(self):
        for i in tqdm(range(self.episodes), desc="Training", unit="iter"):
            episode_reward = 0

            state = self.env.reset()[0]
            state_idx = tuple(state) if isinstance(state, np.ndarray) else (state, )
            action = self.eps_greedy_pick(state_idx)
            terminated = False
            truncated = False
            while (not terminated and not truncated):
                new_state, reward, terminated, truncated, info = self.env.step(action)

                state_idx = tuple(state) if isinstance(state, np.ndarray) else (state, )
                new_state_idx = tuple(new_state) if isinstance(new_state, np.ndarray) else (new_state, )
                action_state_idx = state_idx + (action,)

                new_action = self.eps_greedy_pick(new_state_idx)
                new_action_state_idx = new_state_idx + (new_action,)
                self.q[action_state_idx] += self.lr * (reward + self.df * self.q[new_action_state_idx] - self.q[action_state_idx])
                self.pi[state_idx] = np.argmax(self.q[state_idx])
                state = new_state
                action = new_action
                
                episode_reward += reward
                
            self.eps = max(self.eps - self.eps_decay_rate, 0)
            self.rewards[i] = episode_reward

        # plot stuff
        env_name = self.env.spec.id
        s = env_name + "_" + str(self.episodes) + "_episodes_sarsa"
        path = "reward_" + s + ".png"
        cumulative_reward = plot_cumulative_reward(self.episodes, self.rewards, path, "Sarsa")

        # save stuff
        name = "learned_q_" + s + ".npy"
        np.save(name, self.q)
        name = "learned_policy_" + s + ".npy"
        np.save(name, self.pi)

        if self.save_cumulative_reward:
            name = "cumulative_reward_" + s + ".npy"
            np.save(name, cumulative_reward)