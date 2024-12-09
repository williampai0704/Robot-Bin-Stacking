import numpy as np
import pandas as pd

def read_csv(infile):
    df = pd.read_csv(infile)
    # reward = np.array(df["reward"].values)
    # efficiency = np.array(df['efficiency'].values)
    # collision = np.array(df['collision_penalty'].values)
    # stack = np.array(df['stack_penalty'].values)

    rewards = []
    effs = [] 
    num_coll = 0.

    for i in range(0, len(df), 2):  # Step by 2
        if i + 1 < len(df):
            reward = df.loc[i, "reward"]
            # print(reward)
            reward2 = df.loc[i+1, "reward"]
            total_reward = reward + reward2
            rewards.append(total_reward)

            # eff = df.loc[i, "efficiency"]
            eff2 = df.loc[i+1, "efficiency"]
            # total_eff = eff + eff2
            effs.append(eff2)

            
            if df.loc[i, "collision_penalty"] != 0:
                num_coll += 1
            if df.loc[i + 1, "collision_penalty"] != 0:
                num_coll += 1
    
    rewards = np.array(rewards)
    effs = np.array(effs)
    idx = np.argsort(rewards)[::-1]

    print(f"collision percent: {num_coll / len(df)}")
    print(f"Top 10 rewards: {rewards[idx][:10]}")
    print(f"Corresponding efficiencies: {effs[idx][:10]}")


read_csv("inference_100_model.pt.csv")