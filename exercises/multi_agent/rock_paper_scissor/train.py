#!/usr/bin/env python
from distributed_q_learning import distributed_q_learning
from parallel_env import parallel_env as pe

def main():
    num_episodes = 1000
    env = pe(render_mode=None)
    observations, infos = env.reset()
    params ={'eps' : 1,
             'discount_factor' : 1,
             'eps_decay_rate' : 1 / num_episodes
            }
    
    trainer = distributed_q_learning(env, num_episodes, params)
    trainer.run()
    env.close()   

if __name__== '__main__':
    main()