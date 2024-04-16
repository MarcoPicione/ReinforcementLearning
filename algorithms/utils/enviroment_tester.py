import numpy as np
import time

class enviroment_tester():
    def __init__(self, env, q_table = None, policy = None, seed = None, time_sleep = 0.01, verbose = False, visualize = True):
        if q_table is None and policy is None:
            raise Exception("q_table or policy must be not none")
        self.env = env
        self.q = q_table
        self.policy = policy
        self.verbose = verbose
        self.seed = seed
        self.time_sleep = time_sleep
        self.visualize = visualize

        env.unwrapped.render_mode = None if not visualize else "human"

    def test(self):
        state = self.env.reset(seed = self.seed)[0]
        terminated = False
        reward_tot = 0
        while not terminated:
            state_idx = tuple(state) if isinstance(state, np.ndarray) else (state, )
            action = np.argmax(self.q[state_idx]) if self.q is not None else self.policy[state_idx]
            if self.verbose: print("Action ", action)
            new_state, reward, terminated, _, info = self.env.step(action)
            reward_tot += reward
            if self.visualize: time.sleep(self.time_sleep)
            if self.verbose: print("New state ", new_state)
            if not terminated: state = new_state

        return state, reward_tot