
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import pandas as pd
import ipdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def shuffle_and_sample(self, batch_size):
        """Shuffle the memory in place"""
        # Convert deque to list, shuffle, and convert back to deque
        memory_list = list(self.memory)
        random.shuffle(memory_list)

        batches = [
            memory_list[i:i + batch_size] for i in range(0, len(memory_list), batch_size)
        ]
        return batches

    def __len__(self):
        return len(self.memory)
    

#action plane params
RESOLUTION = 0.1

# GAMMA is the discount factor 
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 1
TAU = 0.005
LR = 1e-4


n_actions = 11**2 ### must check but for now: 0, 0.1, ..., 0.9, 1
state_dim = 10 # 
 
policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


def process_data():

    df = pd.read_csv("stacking_samples_mixed_10000_p0.2.csv")
    df = df[:20]

    state_headers = ["box_0_x","box_0_z","box_0_l","box_0_h","box_c_l","box_c_h","box_1_x","box_1_z","box_1_l","box_1_h"]
    action_headers = ["a_x","a_z"]
    reward_headers = ["reward"]

    # Process the DataFrame two rows at a time
    for i in range(0, len(df), 2):  # Step by 2
        if i + 1 < len(df):  # Ensure there's a next line for the pair
            # Extract state, action, and next_state (2nd box placement, even rows)
            state = df.loc[i, state_headers].values.tolist()
            action = df.loc[i, action_headers].values.tolist()
            next_state = df.loc[i + 1, state_headers].values.tolist()
            reward = df.loc[i, reward_headers].values.tolist()

            memory.push(torch.tensor(state), torch.tensor(action), torch.tensor(next_state), torch.tensor(reward))

            # Extract state, action, and next_state (3rd box placement, odd rows)
            state = df.loc[i + 1, state_headers].values.tolist()
            action = df.loc[i + 1, action_headers].values.tolist()
            # next_state fill with None 
            reward = df.loc[i + 1, reward_headers].values.tolist()

            memory.push(torch.tensor(state), torch.tensor(action), None, torch.tensor(reward))


    return memory 



steps_done = 0



def coord2ind(action, low=0., high=1.):
    width = high - low
    num = int(width/RESOLUTION)

    rounded_action = np.round(action / RESOLUTION) * RESOLUTION
    indx = (((rounded_action[:,0] - low) / width) * num).int() # 0 to num
    indz = (((rounded_action[:,1] - low) / width) * num).int() # 0 to num

    return np.ravel_multi_index([indx, indz], (num + 1 , num + 1)) 

def ind2coord(action_index, low=0., high=1.):
    
    width = high - low
    num = int(width/RESOLUTION)
    coord = np.unravel_index(action_index, (num+1, num+1))
    ax = (coord[0] / num) * width + low
    az = (coord[1] / num) * width + low
    return np.array([ax, az]).T # batch, 2




def optimize_model(transitions):
    # if len(memory) < BATCH_SIZE:
    #     return
    # transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # transitions = memory.sample(BATCH_SIZE)
    # batch = Transition(*zip(*transitions))
    # print(batch)


    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

    # print(batch.state)

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    action_batch_idx = torch.from_numpy(coord2ind(action_batch))
    
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(policy_net(state_batch))
    # print("net output: \n", policy_net(state_batch).shape)
    # ipdb.set_trace()
    state_action_values = policy_net(state_batch).gather(1, action_batch_idx.unsqueeze(-1))
    # print("state_action_values:\n", state_action_values)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(state_batch.shape[0], device=device)
    with torch.no_grad():
        # print(next_state_values[non_final_mask].shape)
        # print(target_net(non_final_next_states).shape)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def main():
    num_epochs = 10000

    memory = process_data()
    # print(memory.sample(5))
    for i_epoch in range(num_epochs):
        batches = memory.shuffle_and_sample(BATCH_SIZE)
        for batch in batches:
            # print(batch)

            optimize_model(batch)
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

    print('Complete')
    torch.save(target_net.state_dict(), "model.pt")

if __name__ == "__main__":
    main()

