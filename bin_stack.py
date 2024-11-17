import numpy as np
import argparse
import pybullet as p
from matplotlib import pyplot as plt
from bp_env import BinStackEnviornment

def main():
    
    env = BinStackEnviornment(True)
    box_id, dim = env.load_box()
    print("box dimension = " + str(dim))
    succ = env.execute_grasp(box_id,dim)
    env.move_tool([5,5,2], p.getQuaternionFromEuler([np.pi, 0, np.pi/2]))
    while True:
        env.step_simulation(1)
    
if __name__ == '__main__':
    main()