#!/usr/bin/env python
from parallel_env import parallel_env as pe
import parallel_env
import numpy as np
import pickle

def map(n):
     if n == 0: return "ROCK"
     if n == 1: return "PAPER"
     return "SCISSOR"
def main():
    name ="learned_qs_rsp_1000_episodes_.pkl"
    with open(name, 'rb') as f:
            q = pickle.load(f)
    print(q)

    env = pe(render_mode="human")
    observations, infos = env.reset()

    for i in range(10): print('\n')

    while env.agents:
        actions = {a: np.argmax(q[a][observations[a]]) for a in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print("Actions: ", {a: map(actions[a]) for a in env.agents})

if __name__== '__main__':
    main()