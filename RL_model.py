
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
from tqdm import trange

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        # Input layer
        self.layer1 = nn.Linear(n_observations, 512)
        self.bn1 = nn.BatchNorm1d(512)  # Batch normalization
        
        # Hidden layers
        self.layer2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.layer4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Output layer
        self.layer5 = nn.Linear(256, n_actions)
        
        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # Pass through the network with activations, batch normalization, and dropout
        x = F.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.layer3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.layer4(x)))
        return self.layer5(x)  # No activation for the output, as it's used directly in Q-learning

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
RESOLUTION = 0.05

# GAMMA is the discount factor 
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 1
TAU = 0.02
LR = 2e-5
LOWx, HIx = -0.5, 0.5
LOWz, HIz = 0., 1.

# n_actions = 101**2 ### must check but for now: 0, 0.1, ..., 0.9, 1
n_actions = int(1 / RESOLUTION + 1)**2
state_dim = 10 # 
 
policy_net = DQN(state_dim, n_actions).to(device)
target_net = DQN(state_dim, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(60000)
print(len(memory))

def process_data(datacsv):

    # df = pd.read_csv("stacking_samples_mixed_p1.0.csv")
    df = pd.read_csv(datacsv)
    # df = df[:20]

    state_headers = ["box_0_x","box_0_z","box_0_l","box_0_h","box_c_l","box_c_h","box_1_x","box_1_z","box_1_l","box_1_h"]
    action_headers = ["a_x","a_z"]
    reward_headers = ["reward"]
    stack_penalty_headers = ["stack_penalty"]
    collision_headers = ["collision_penalty"]

    # Process the DataFrame two rows at a time
    for i in range(0, len(df), 2):  # Step by 2
        if i + 1 < len(df):  # Ensure there's a next line for the pair
            # Extract state, action, and next_state (2nd box placement, even rows)
            state = df.loc[i, state_headers].values.tolist()
            action = df.loc[i, action_headers].values.tolist()
            # action = np.clip(action, 0, 1)
            next_state = df.loc[i + 1, state_headers].values.tolist()

            # notstacked = df.loc[i, stack_penalty_headers].values.tolist()
            reward = df.loc[i, reward_headers].values.tolist()
            
            # collision = df.loc[i, collision_headers].values.tolist()
            # reward = - np.array(collision)*0.7*0.2 + np.array(reward) ## collision is -100*0.7 -> -70; add 70 to reward so it's effectively smt like -30

            memory.push(torch.tensor(state), torch.tensor(action), torch.tensor(next_state), torch.tensor(reward))

            # Extract state, action, and next_state (3rd box placement, odd rows)
            state = df.loc[i + 1, state_headers].values.tolist()
            action = df.loc[i + 1, action_headers].values.tolist()
            # action = np.clip(action, 0, 1)
            # next_state fill with None 
            # notstacked = df.loc[i + 1, stack_penalty_headers].values.tolist()
            reward = df.loc[i + 1, reward_headers].values.tolist()
            # reward = np.array(notstacked)*0.2 + np.array(reward) ## increase stack penalty
            # collision = df.loc[i + 1, collision_headers].values.tolist()
            # reward = - np.array(collision)*0.7*0.2 + np.array(reward)

            memory.push(torch.tensor(state), torch.tensor(action), None, torch.tensor(reward))


    return memory 


def round_to_res(action):
    return np.round(action / RESOLUTION) * RESOLUTION


def coord2ind(action):
    # equal width x and z
    width = HIx - LOWx
    num = int(width/RESOLUTION)

    rounded_action = np.round(action / RESOLUTION) * RESOLUTION

    indx = (((rounded_action[:,0] - LOWx) / width) * num).int() # 0 to num
    indx = np.clip(indx, 0, num)
    indz = (((rounded_action[:,1] - LOWz) / width) * num).int() # 0 to num
    indz = np.clip(indx, 0, num)

    # return np.ravel_multi_index([indx, indz], (num + 1 , num + 1)) 
    try:
        return np.ravel_multi_index([indx, indz], (num + 1, num + 1))
    except Exception as e:
        print("Error occurred in np.ravel_multi_index!")
        print("Inputs:")
        print(f"indx: {indx}")
        print(f"indz: {indz}")
        ipdb.set_trace()

def ind2coord(action_index):
    
    width = HIx - LOWx
    num = int(width/RESOLUTION)
    coord = np.unravel_index(action_index, (num+1, num+1))
    ax = (coord[0] / num) * width + LOWx
    az = (coord[1] / num) * width + LOWz
    return np.array([ax, az]).T # batch, 2


def optimize_model(transitions):
    # if len(memory) < BATCH_SIZE:
    #     return
    # transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device)

    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.stack(batch.action)

    action_batch_idx = torch.from_numpy(coord2ind(action_batch)).to(device)

    reward_batch = torch.stack(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    state_action_values = policy_net(state_batch).gather(1, action_batch_idx.unsqueeze(-1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(state_batch.shape[0], device=device)
    with torch.no_grad():

        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(-1) * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss


def test(model):
    df = pd.read_csv("train_data/train_5000_p1.0_unnoised.csv")
    df = df[:100]
    opt_actions = np.array(df[["a_x","a_z"]].values)
    opt_actions = round_to_res(opt_actions) 
    # opt_actions = np.clip(opt_actions, 0, 1)
    state_headers = ["box_0_x","box_0_z","box_0_l","box_0_h","box_c_l","box_c_h","box_1_x","box_1_z","box_1_l","box_1_h"]

    state = df[state_headers].values.tolist()
    # state = np.clip(state, 0, 1)
    ex_states = torch.tensor(state, dtype=torch.float32).to(device)

    pred_actions = model(ex_states).argmax(1).cpu().numpy()
    act_coords = ind2coord(pred_actions)
    # print(act_coords[:5].T, opt_actions[:5].T)
    print(np.mean((act_coords-opt_actions)**2), np.mean(np.abs(act_coords-opt_actions)))


if __name__ == "__main__":
    num_epochs = 400 #000

    trainfile = "new_train_data/train_53607_pmixed_nmixed_r0.05.csv"
    memory = process_data(trainfile)
    
    # trainfile = "train_data/train_5000_p1.0_unnoised.csv"
    # memory = process_data(trainfile)
    
    # trainfile = "train_data/train_5000_p1.0_unnoised_2.csv"
    # memory = process_data(trainfile)
    
    # trainfile = "train_data/train_5000_p0.0_unnoised_r0.05.csv"
    # memory = process_data(trainfile)
    
    print(len(memory))

    losses = []
    for i_epoch in trange(num_epochs, desc="Training Epochs"):
        batches = memory.shuffle_and_sample(BATCH_SIZE)
        # epoch_loss = 0
        for ii, batch in enumerate(batches):
            # print(batch)

            loss = optimize_model(batch)
            
            # epoch_loss += loss
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            # if (ii % int(len(batches)/ 5)) == 0:
            #     policy_net.eval()
            #     test(policy_net)
            #     target_net.eval()
            #     test(target_net)
        # losses.append(epoch_loss / len(batches))
        # print(losses[-1])
        if (i_epoch % 5) == 0:
            print(i_epoch)

            # torch.save(target_net.state_dict(), "target_net.pt")
            # torch.save(policy_net.state_dict(), "policy_net.pt")
            policy_net.eval()
            test(policy_net)
            target_net.eval()
            test(target_net)


    print('Complete')
    torch.save(target_net.state_dict(), f"model_r{RESOLUTION}.pt")

    # plt.figure()
    # plt.plot(np.arange(num_epochs), losses.cpu())
    # plt.title("loss vs epoch")
    # plt.savefig("loss.png")




