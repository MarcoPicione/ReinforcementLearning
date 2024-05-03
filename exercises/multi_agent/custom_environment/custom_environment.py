import functools
import random
from random import seed
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import gymnasium as gym

from pettingzoo import ParallelEnv


class CustomEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, render_mode = None):
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]
        self.observation_spaces = {a:  MultiDiscrete([7 * 7 - 1] * 3) for a in self.possible_agents}
        self.action_spaces = {a:  Discrete(4) for a in self.possible_agents}
        self.render_mode = render_mode


    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner_x = 0
        self.prisoner_y = 0

        self.guard_x = 6
        self.guard_y = 6

        self.escape_x = random.randint(2, 5)
        self.escape_y = random.randint(2, 5)

        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )
        observations = {
            "prisoner": {"observation": observation, "action_mask": [0, 1, 1, 0]},
            "guard": {"observation": observation, "action_mask": [1, 0, 0, 1]},
        }

        self.rewards = {a: 0 for a in self.agents}

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}
        
        # Execute actions
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]

        if prisoner_action == 0 and self.prisoner_x > 0:
            self.prisoner_x -= 1
        elif prisoner_action == 1 and self.prisoner_x < 6:
            self.prisoner_x += 1
        elif prisoner_action == 2 and self.prisoner_y > 0:
            self.prisoner_y -= 1
        elif prisoner_action == 3 and self.prisoner_y < 6:
            self.prisoner_y += 1

        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < 6:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < 6:
            self.guard_y += 1

        # Generate action masks
        prisoner_action_mask = np.ones(4, dtype=np.int8)
        if self.prisoner_x == 0:
            prisoner_action_mask[0] = 0  # Block left movement
        elif self.prisoner_x == 6:
            prisoner_action_mask[1] = 0  # Block right movement
        if self.prisoner_y == 0:
            prisoner_action_mask[2] = 0  # Block down movement
        elif self.prisoner_y == 6:
            prisoner_action_mask[3] = 0  # Block up movement

        guard_action_mask = np.ones(4, dtype=np.int8)
        if self.guard_x == 0:
            guard_action_mask[0] = 0
        elif self.guard_x == 6:
            guard_action_mask[1] = 0
        if self.guard_y == 0:
            guard_action_mask[2] = 0
        elif self.guard_y == 6:
            guard_action_mask[3] = 0

        # Action mask to prevent guard from going over escape cell
        if self.guard_x - 1 == self.escape_x:
            guard_action_mask[0] = 0
        elif self.guard_x + 1 == self.escape_x:
            guard_action_mask[1] = 0
        if self.guard_y - 1 == self.escape_y:
            guard_action_mask[2] = 0
        elif self.guard_y + 1 == self.escape_y:
            guard_action_mask[3] = 0

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        # Check truncation conditions (overwrites termination conditions)
        truncations = {"prisoner": False, "guard": False}
        if self.timestep > 100:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
            self.agents = []
        self.timestep += 1

        # Get observations
        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )
        observations = {
            "prisoner": {
                "observation": observation,
                "action_mask": prisoner_action_mask,
            },
            "guard": {"observation": observation, "action_mask": guard_action_mask},
        }

        # Get dummy infos (not used in this example)
        infos = {"prisoner": {}, "guard": {}}

        self.rewards = copy(rewards)
        self.terminations = copy(terminations)
        self.truncations = copy(truncations)

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return
        elif self.render_mode == "human":
            # grid = np.full((7, 7), "_")
            # grid[self.prisoner_y, self.prisoner_x] = "P"
            # grid[self.guard_y, self.guard_x] = "G"
            # grid[self.escape_y, self.escape_x] = "E"
            # print("\033c")
            # print(f"{grid} \n")

            print("\033c")
            for r in range(7):
                for c in range(7):

                    if([r,c] == [self.prisoner_y, self.prisoner_x]):
                        print("P", end=' ')
                    elif([r,c] == [self.guard_y, self.guard_x]):
                        print("G", end=' ')
                    elif([r,c] == [self.escape_y, self.escape_x]):
                        print("E", end=' ')
                    else:
                        print("_", end=' ')

                print()
            print()


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def observe(self, agent):
        return np.array(self.observations[agent])
    
    def seed(seed=None):
        seed(seed)

    def close(self):
        pass

    def state(self):
        pass

    def state_space(self):
        pass

