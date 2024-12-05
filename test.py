import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd


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
    

model = DQN(10,11**2)
model.load_state_dict(torch.load("model.pt", weights_only=True))
model.eval()

df = pd.read_csv("stacking_samples_mixed_p1.0.csv")
opt_actions = torch.tensor(df[["a_x","a_z"]].values, dtype=torch.float32)
state_headers = ["box_0_x","box_0_z","box_0_l","box_0_h","box_c_l","box_c_h","box_1_x","box_1_z","box_1_l","box_1_h"]

ex_states = torch.tensor(df[state_headers].values, dtype=torch.float32)

pred_actions = model(ex_states).argmax(1)

def ind2coord(action_index, low=0., high=1.):
    
    width = high - low
    num = int(width/0.1)
    coord = np.unravel_index(action_index, (num+1, num+1))
    ax = (coord[0] / num) * width + low
    az = (coord[1] / num) * width + low
    return np.array([ax, az]).T # batch, 2

print(pred_actions)

print(ind2coord(pred_actions))
print(opt_actions)