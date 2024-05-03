import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithms.utils.plots import plot_cumulative_reward
import pickle
# import functools

class distributed_q_learning():
    def __init__(self, env, episodes, params, save_cumulative_reward = False):
        self.env = env
        self.agents = env.agents[:]
        self.episodes = episodes
        self.eps = params['eps']
        # self.lr = params['learning_rate']
        self.df = params['discount_factor']
        self.eps_decay_rate = params['eps_decay_rate']
        self.save_cumulative_reward = save_cumulative_reward
        self.rng = np.random.default_rng() 
        self.q = {agent : np.zeros((self.env.observation_space(agent).n, self.env.action_space(agent).n)) for agent in self.agents}
        self.rewards = [{} for i in range(self.episodes)]

    def eps_greedy_pick(self, observations):
        actions = {}
        for k in observations.keys():
            state = observations[k]
            if self.rng.random() < self.eps:
                action = self.env.action_space(k).sample()
            else:
                action = np.argmax(self.q[k][state])
            actions[k] = action
        return actions
    
    # @functools.lru_cache(maxsize=None)
    def improve_q_function(self, agent, state, action, reward, new_state):
        new_value = reward + self.df * np.max(self.q[agent][new_state])
        self.q[agent][state, action] = np.max([self.q[agent][state, action], new_value])
        # self.q[agent][state, action] += 0.1 * (reward + self.df * np.max(self.q[agent][new_state]) - self.q[agent][state, action])

    def run(self):
        for i in tqdm(range(self.episodes), desc="Training", unit="iter"):
            episode_rewards = {agent: 0 for agent in self.agents}

            observations = self.env.reset()[0]
            terminations = {agent: False for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            while (not all(terminations.values()) and not all(truncations.values())):
                actions = self.eps_greedy_pick(observations)
                new_observations, rewards, terminations, truncations, infos = self.env.step(actions)
                for a in self.agents: self.improve_q_function(a, observations[a], actions[a], rewards[a], new_observations[a])
                
                observations = new_observations.copy()
                for k in episode_rewards.keys():
                    episode_rewards[k] += rewards[k]
                
            self.eps = max(self.eps - self.eps_decay_rate, 0)
            self.rewards[i] = episode_rewards.copy()

        # plot stuff
        # env_name = self.env.spec.id
        env_name = "rsp"
        s = env_name + "_" + str(self.episodes) + "_episodes_"
        # path = "reward_" + s + ".png"
        # cumulative_reward = plot_cumulative_reward(self.episodes, self.rewards, path, "q_learning")

        # save stuff
        name = "learned_qs_" + s + ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(self.q, f)

        # if self.save_cumulative_reward:
        #     name = "cumulative_reward_" + s + ".npy"
        #     np.save(name, cumulative_reward)