import time
import torch

class enviroment_tester_nn():
    def __init__(self, env, q_network, seed = None, time_sleep = 0.01, verbose = False, visualize = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.q = q_network.to(self.device)
        self.q.to(self.device)
        self.verbose = verbose
        self.seed = seed
        self.time_sleep = time_sleep
        self.visualize = visualize

        env.unwrapped.render_mode = None if not visualize else "human"

    def test(self):
        state = self.env.reset(seed = self.seed)[0]
        state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)
        terminated = False
        reward_tot = 0
        self.q.eval()

        with torch.no_grad():
            while not terminated:
                action = self.q(state).max(1).indices.item()
                if self.verbose: print("Action ", action)
                next_state, reward, terminated, _, _ = self.env.step(action)
                reward_tot += reward
                if self.visualize: time.sleep(self.time_sleep)
                if self.verbose: print("New state ", next_state)
                if not terminated: state = torch.tensor(next_state, dtype = torch.float32, device = self.device).unsqueeze(0)

        return state, reward_tot