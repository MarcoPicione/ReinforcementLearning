#!/usr/bin/env python
from vacuum_cleaner import Vacuum_cleaner, VacuumCleanerAction
import gymnasium as gym
from pettingzoo.utils import agent_selector
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
import numpy as np

register(
    id = "snake-v0",
    entry_point = "snake_env:snake_env",
)

class vacuum_cleaner_env(gym.Env):
    metadata = {"render_modes" : ["human"], "render_fps" : 1}

    def __init__(self, rows, cols, num_agents, render_mode = None, max_iter = 1000, seed = None):
        self.rows = rows
        self.cols = cols
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.max_iter = max_iter

        bounds = {'x_min' : 1,
                  'x_max': self.cols,
                  'y_min' : 1,
                  'y_max': self.rows
                }
        
        # possible_names = ['A', 'B', 'C', 'D', 'E']
        self.cleaners = [Vacuum_cleaner(str(i), bounds, seed = seed) for i in range(self.num_agents)]
        self.agents = [c.__str__() for c in self.cleaners]

        nA = len(VacuumCleanerAction)
        nS = self.rows * self.cols
        self.observation_space = dict(zip(self.agents, [gym.spaces.Discrete(nS)] * self.num_agents))
        self.action_spaces = dict(zip(self.agents, [gym.spaces.Discrete(nA)] * self.num_agents))

        # Build obstacles
        self.obstacles = [[0, i] for i in range (self.cols)] + [[self.rows - 1, i] for i in range (self.cols)] + \
                         [[i, 0] for i in range (1, self.rows - 1)] + [[i, self.cols - 1] for i in range (1, self.rows - 1)]
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agents = self.possible_agents[:] #????

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        if(self.render_mode=='human'):
            self.render()

        # return obs, info

    def step(self, action):
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0 #??????
        self.state[self.agent_selection] = action

        if self._agent_selector.is_last():
            self.rewards = {a:a.reward for a in self.agents}

            self.num_moves += 1
            self.truncations = {
                agent: self.num_moves >= self.max_iter for agent in self.agents
            }

        else:
            # necessary so that observe() returns a reasonable observation at all times.
            self.state[self.agents[1 - self.agent_name_mapping[agent]]] = None
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
    
    def observe(self, agent):
        return self.agents[agent].state #?????????
    
    def _accumulate_rewards(self):

        
    

    
    def render(self):
        agents_pos = [a.position for a in self.agents]
        visited_cells = set()
        for a in self.agents:
            visited_cells.union(a.visited_cells)

        print("\033c")
        for r in range(self.rows):
            for c in range(self.cols):

                if([r,c] in agents_pos):
                    print('A', end=' ')
                elif([r,c] in visited_cells):
                    print('+', end=' ')
                elif([r,c] in self.obstacles):
                    print('O', end=' ')
                else:
                    print('_', end=' ')

            print()
        print()
    
def main():
    env = gym.make("snake-v0", rows=12, cols=12, render_mode=None)
    print("Check env begin")
    print(env.unwrapped)
    check_env(env.unwrapped)
    print("Check env end")
    
    # Sometimes is deterministic other times not
    # I do not understand

if __name__== '__main__':
    main()



    