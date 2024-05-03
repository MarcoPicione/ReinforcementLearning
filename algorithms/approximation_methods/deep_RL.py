import math
import random
from matplotlib import pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

"""
STEPS:
    1) Initialize the q network with random weights
    2) Sample an action using a eps greedy policy
    3) Execute the action and observe next state and next reward
    4) Store the experience tuple (state, action, reward, next_state) in a replay buffer
    5) Sample a mini-batch of experience from the replay buffer
    6) Compute the Q-target values from the mini-batch Q_t = r + gamma * max_a[Q(next_state, a)]
    7) Compute the Q values for the mini-batch using the current Q network
    8) Compute the loss between the Q-values and the Q-target values and update the network parameters
    9) Repeat 2-8 for a fixed number of episodes
"""

class deep_RL:
    def __init__(self, parameters, network_class, env, num_episodes):

        self.batch_size = parameters['batch_size']
        self.gamma = parameters['gamma']
        self.eps_start = parameters['eps_start']
        self.eps_end = parameters['eps_end']
        self.eps_decay = parameters['eps_decay']
        self.tau = parameters['tau']
        self.lr = parameters['lr']
        self.memory_size = parameters['memory_size']

        self.memory = ReplayMemory(self.memory_size)
        self.num_episodes = num_episodes
        self.env = env

        nS = len(self.env.reset()[0])
        nA = self.env.unwrapped.action_space.n
        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = network_class(nS, nA).to(self.device)
        self.target_net = network_class(nS, nA).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.lr, amsgrad = True)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device = self.device, dtype = torch.long)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch) #????

        next_state_values = torch.zeros(self.batch_size, device = self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100) #???
        self.optimizer.step()

    def run(self):
        best_reward = -np.inf
        for i in tqdm(range(self.num_episodes), desc="Training", unit="iter"):
            episode_reward = 0
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                episode_reward += reward
                reward = torch.tensor([reward], device = self.device)
                done = terminated or truncated
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype = torch.float32, device = self.device).unsqueeze(0)

                self.memory.push(state, action, reward, next_state)

                state = next_state
                
                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 − τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(episode_reward)
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        with open('saved_dicts/policy_net_dict_reward_' + str(episode_reward) + '.pkl', 'wb') as f:
                            pickle.dump(self.policy_net.state_dict(), f)

                    # self.plot_durations()
                    break

        self.plot_durations()

        with open('policy_net_dict.pkl', 'wb') as f:
            pickle.dump(self.policy_net.state_dict(), f)


    def plot_durations(self):
        plt.figure()
        plt.title('Episode duration during training')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(self.episode_durations)
        plt.show()

    # def plot_durations(self, show_result=False):
    #     plt.figure(1)
    #     durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    #     if show_result:
    #         plt.title('Result')
    #     else:
    #         plt.clf()
    #         plt.title('Training...')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Duration')
    #     plt.plot(durations_t.numpy())
    #     # Take 100 episode averages and plot them too
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())

    #     plt.pause(0.001)  # pause a bit so that plots are updated
"""
We' ll be using experience replay memory for training our DQN. It stores the transitions that the agent 
observes, allowing us to reuse this data later. By sampling from it randomly, the transitions that build 
up a batch are decorrelated. It has been shown that this greatly stabilizes and improves the DQN training 
procedure.
For this, we’re going to need two classes:

Transition - a named tuple representing a single transition in our environment. It essentially maps (state, action) 
pairs to their (next_state, reward) result, with the state being the screen difference image as described later on.

ReplayMemory - a cyclic buffer of bounded size that holds the transitions observed recently. It also implements 
a .sample() method for selecting a random batch of transitions for training.
"""

Transition = namedtuple('Transition',('state','action','reward','next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)