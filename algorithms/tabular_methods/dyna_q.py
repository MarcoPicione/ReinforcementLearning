import numpy as np
from tqdm import tqdm
from algorithms.utils.plots import plot_cumulative_reward
import random

class dyna_q():
    def __init__(self, env, episodes, params, planning_steps):
        self.env = env
        self.episodes = episodes
        self.eps = params['eps']
        self.lr = params['learning_rate']
        self.df = params['discount_factor']
        self.eps_decay_rate = params['eps_decay_rate']
        self.n = planning_steps
        self.rng = np.random.default_rng()
        self.nS = self.env.observation_space.n
        self.nA = self.env.action_space.n
        self.q = np.zeros((self.nS, self.nA))
        self.model = dict() #{s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
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
                action = self.eps_greedy_pick(state)

                new_state, reward, terminated, truncated, info = self.env.step(action)

                state_action_idx = (state, action)

                self.q[state_action_idx] += self.lr * (reward + self.df * np.max(self.q[new_state]) - self.q[state_action_idx])
                if state in self.model.keys():
                        self.model[state][action] = (reward, new_state)
                else:
                    self.model[state] = {action: (reward, new_state)}
                

                state = new_state
                episode_reward += reward

                for _ in range(self.n):
                    s = random.choice(list(self.model.keys()))
                    a = random.choice(list(self.model[s].keys()))

                    r, new_s = self.model[s][a]
                    self.q[s, a] += self.lr * (r + self.df * np.max(self.q[new_s]) - self.q[s, a])
                
            self.eps = max(self.eps - self.eps_decay_rate, 0)
            self.rewards[i] = episode_reward

        # plot stuff
        env_name = self.env.spec.id
        s = env_name + "_" + str(self.episodes) + "_episodes_" + str(self.n) + "_step_planning_dyna_q"
        path = "reward_" + s + ".png"
        plot_cumulative_reward(self.episodes, self.rewards, path, "dyna_q_" + str(self.n) + "_planning")

        # save stuff
        name = "learned_q_" + s + ".npy"
        np.save(name, self.q)