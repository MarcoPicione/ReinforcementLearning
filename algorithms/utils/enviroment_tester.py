import numpy as np
import time

class enviroment_tester():
    def __init__(self, env, q_table, seed = None, time_sleep = 0.01, verbose = False):
        self.env = env
        self.q = q_table
        self.verbose = verbose
        self.seed = seed
        self.time_sleep = time_sleep

    def test(self):
        state = self.env.reset(seed = self.seed)[0]
        terminated = False
        reward_tot = 0
        while not terminated:
            state_idx = tuple(state) if isinstance(state, np.ndarray) else (state, )
            action = np.argmax(self.q[state_idx])
            if self.verbose: print("Action ", action)
            new_state, reward, terminated, _, info = self.env.step(action)
            reward_tot += reward
            time.sleep(0.01)
            if self.verbose: print("New state ", new_state)
            if not terminated: state = new_state

        return state, reward_tot