
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

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.dropout = nn.Dropout(0.2) 
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 512)
        self.layer4 = nn.Linear(512, 256)
        self.layer5 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.layer5(x)
    


Transition = namedtuple('Transition',
                        ('state', 'action'))


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
TAU = 0.01
LR = 1e-4
LOWx, HIx = -0.5, 0.5
LOWz, HIz = 0., 1.

n_actions = 21**2 ### must check but for now: 0, 0.1, ..., 0.9, 1
state_dim = 10 # 
 
policy_net = DQN(state_dim, n_actions).to(device)



optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(50000)
print(len(memory))

def process_data(datacsv):

    # df = pd.read_csv("stacking_samples_mixed_p1.0.csv")
    df = pd.read_csv(datacsv)
    # df = df[:20]

    state_headers = ["box_0_x","box_0_z","box_0_l","box_0_h","box_c_l","box_c_h","box_1_x","box_1_z","box_1_l","box_1_h"]
    action_headers = ["a_x","a_z"]


    # Process the DataFrame two rows at a time
    for i in range(0, len(df), 2):  # Step by 2
        if i + 1 < len(df):  # Ensure there's a next line for the pair
            # Extract state, action, and next_state (2nd box placement, even rows)
            state = df.loc[i, state_headers].values.tolist()
            action = df.loc[i, action_headers].values.tolist()
            
            memory.push(torch.tensor(state), torch.tensor(action))

            # Extract state, action (3rd box placement, odd rows)
            state = df.loc[i + 1, state_headers].values.tolist()
            action = df.loc[i + 1, action_headers].values.tolist()

            memory.push(torch.tensor(state), torch.tensor(action))


    return memory 





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

    batch = Transition(*zip(*transitions))



    state_batch = torch.stack(batch.state).to(device)
    action_batch = torch.stack(batch.action)

    action_batch_idx = torch.from_numpy(coord2ind(action_batch)).to(device) # target action classes
    # print(action_batch_idx.shape)

    state_action_logits = policy_net(state_batch)  # logits #.argmax(1) # predicted indices
    # print(state_action_logits.shape)

    # Compute Huber loss
    criterion = nn.CrossEntropyLoss()

    loss = criterion(state_action_logits, action_batch_idx)

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
    opt_actions = np.round(opt_actions, 2) # 2 decimals, same as resolution 0.01
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
    num_epochs = 50 #000

    # trainfile = "train_data/train_10000_p0.0_noised.csv"
    # memory = process_data(trainfile)

    # trainfile = "train_data/train_10000_p1.0_noised.csv"
    # memory = process_data(trainfile)

    trainfile = "train_data/train_5000_p1.0_unnoised.csv"
    memory = process_data(trainfile)

    trainfile = "train_data/train_5000_p1.0_unnoised_2.csv"
    memory = process_data(trainfile)
    print(len(memory))

    losses = []
    for i_epoch in trange(num_epochs, desc="Training Epochs"):
        batches = memory.shuffle_and_sample(BATCH_SIZE)
        epoch_loss = 0
        for ii, batch in enumerate(batches):
            # print(batch)

            loss = optimize_model(batch)
            
            epoch_loss += loss

        losses.append(epoch_loss / len(batches))
        print(losses[-1])
        if (i_epoch % 10) == 0:
            print(i_epoch)
            torch.save(policy_net.state_dict(), f"policy_net_{i_epoch}.pt")
            policy_net.eval()
            test(policy_net)


    print('Complete')
    torch.save(policy_net.state_dict(), "model.pt")

    # plt.figure()
    # plt.plot(np.arange(num_epochs), losses.cpu())
    # plt.title("loss vs epoch")
    # plt.savefig("loss.png")




