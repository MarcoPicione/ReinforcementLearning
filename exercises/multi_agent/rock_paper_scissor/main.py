#!/usr/bin/env python
from pettingzoo.butterfly import cooperative_pong_v5

def main():
    env = cooperative_pong_v5.env(render_mode = "human")
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
    env.close()

if __name__== '__main__':
    main()