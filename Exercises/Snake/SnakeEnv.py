#!/usr/bin/env python
import Snake as s

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register
import numpy as np

def flatten(xss):
    return [x for xs in xss for x in xs]

register(
    id = "snake-v0",
    entry_point = "SnakeEnv:SnakeEnv",
)

class SnakeEnv(gym.Env):
    metadata = {"render_modes" : ["human"], "render_fps" : 1}

    def __init__(self, rows, cols, render_mode = None):
        self.rows = rows
        self.cols = cols
        self.render_mode = render_mode
        self.snake = s.Snake(rows, cols)
        self.food_counter = 0
        self.action_space = spaces.Discrete(len(s.SnakeAction) - 1)

        # Build low and high list
        # l = [0, 0, 0, 0]
        # h = []
        # for i in range(self.snake.snake_body_max):
        #     l += [-1, -1]

        # for i in range(2 + self.snake.snake_body_max):
        #     h += [self.rows-1, self.cols-1]

        self.observation_space = spaces.Box(
            low = 0,
            high=np.ones(11),
            # shape = (4,),
            dtype = np.int8
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # gym requires this call to control randomness and reproduce scenarios.
        self.snake.reset(seed=seed)
        self.food_counter = 0
        obs = observation = self.snake.build_state().astype(np.int8)
        np.concatenate((self.snake.snake_head_pos, self.snake.food_pos, flatten(self.snake.snake_body))).astype(np.int32)
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

        dist_now = np.linalg.norm(np.array(self.snake.snake_head_pos)-np.array(self.snake.food_pos))
        dist_prev = np.linalg.norm(np.array(self.snake.snake_head_pos_prev)-np.array(self.snake.food_pos))
        # if dist_now < dist_prev:
        #     reward = 1
        # else:
        #     reward = -2

        if food_reached:
            reward = 100
            # if(self.food_counter == 10):
            #     terminated = True
        
        if body_encountered:
            # reward = -1000
            terminated = True
            # if(self.render_mode=="human"): 
            #     print("Body collision")
            #     input()

        if obstacle_encountered:
            # reward = -1000
            terminated = True
            # if(self.render_mode=="human"): 
            #     print("Obstacle collision")
            #     input()

        # observation = np.concatenate((self.snake.snake_head_pos, self.snake.food_pos, flatten(self.snake.snake_body))).astype(np.int32)
        observation = self.snake.build_state().astype(np.int8)
        info = {}
        truncated = False

        if(self.render_mode == "human"):
            self.render()

        return observation, reward, terminated, truncated, info
    
def main():
    env = gym.make("snake-v0", rows=12, cols=12, render_mode="human")
    print("Check env begin")
    print(env.unwrapped)
    check_env(env.unwrapped)
    print("Check env end")
    obs = env.reset()[0]
    

    for i in range(10):
        rand_action = env.action_space.sample()
        obs, rew, terminated, trunvcated, info = env.step(rand_action)

    env.unwrapped.snake.add_body() 

    for i in range(10):
        rand_action = env.action_space.sample()
        obs, rew, terminated, trunvcated, info = env.step(rand_action)

    print(SnakeEnv.observation_space.shape)

if __name__== '__main__':
    main()