#!/usr/bin/env python
import exercises.snake.snake as s

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
import numpy as np

register(
    id = "snake-v0",
    entry_point = "snake_env:snake_env",
)

class snake_env(gym.Env):
    metadata = {"render_modes" : ["human"], "render_fps" : 1}

    def __init__(self, rows, cols, render_mode = None):
        self.rows = rows
        self.cols = cols
        self.render_mode = render_mode
        self.snake = s.Snake(rows, cols)
        self.food_counter = 0

        nA = len(s.SnakeAction) - 1
        nS = 2**11
        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.snake.reset(seed=seed)
        self.food_counter = 0
        obs = self.snake.build_state()
        info = {}

        if(self.render_mode=='human'):
            self.render()

        return obs, info

    def render(self):
        self.snake.render()

    def step(self, action):
        food_reached, body_encountered, obstacle_encountered = self.snake.perform_action(s.SnakeAction(action))

        #Reward
        reward = 0
        terminated = False

        if food_reached:
            reward = 100
        
        if body_encountered:
            # reward = -1000
            terminated = True

        if obstacle_encountered:
            # reward = -1000
            terminated = True

        observation = self.snake.build_state()
        info = {}
        truncated = False

        if(self.render_mode == "human"):
            self.render()

        return observation, reward, terminated, truncated, info
    
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