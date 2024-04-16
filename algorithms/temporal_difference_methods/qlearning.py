import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithms.utils.plots import plot_cumulative_reward

class qlearning():
    def __init__(self, env, episodes, params):
        self.env = env
        self.episodes = episodes
        self.eps = params['eps']
        self.lr = params['learning_rate']
        self.df = params['discount_factor']
        self.eps_decay_rate = params['eps_decay_rate']
        self.rng = np.random.default_rng() 
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.rewards = np.zeros(self.episodes)

    def eps_greedy_pick(self, state):
        if self.rng.random() < self.eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q[state])
        return action

    def run(self):
        for i in tqdm(range(self.episodes), desc="Training", unit="iter"):
            episode_reward = 0

            state = self.env.reset()[0]
            terminated = False
            truncated = False
            while (not terminated and not truncated):
                state_idx = tuple(state) if isinstance(state, np.ndarray) else (state, )
                action = self.eps_greedy_pick(state_idx)
                new_state, reward, terminated, truncated, info = self.env.step(action)

                new_state_idx = tuple(new_state) if isinstance(new_state, np.ndarray) else (new_state, )
                action_state_idx = state_idx + (action,)

                self.q[action_state_idx] += self.lr * (reward + self.df * np.max(self.q[new_state_idx]) - self.q[action_state_idx])
                
                state = new_state
                episode_reward += reward
                
            self.eps = max(self.eps - self.eps_decay_rate, 0)
            self.rewards[i] = episode_reward

        # plot stuff
        env_name = self.env.spec.id
        s = env_name + "_" + str(self.episodes) + "_episodes_"
        path = "reward_" + s + ".png"
        plot_cumulative_reward(self.episodes, self.rewards, path, "q_learning")

        # save stuff
        name = "learned_q_" + s + ".npy"
        np.save(name, self.q)