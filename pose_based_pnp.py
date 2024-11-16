import numpy as np
import argparse
from matplotlib import pyplot as plt
from env import UR5PickEnviornment
from preception import pose_est_state, pose_est_segicp


def main():
    # parse input 
    parser = argparse.ArgumentParser(description='Pick and Place with Pose Estimation')
    parser.add_argument('--use_state', action='store_true')
    parser.add_argument('--check_sim', action='store_true')
    args = parser.parse_args()


    objnames = ['YcbTennisBall', 'YcbFoamBrick',  'YcbStrawberry', 'YcbBanana']
    env = UR5PickEnviornment(True)
    env.load_ycb_objects(objnames)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    cnt = -1
    if args.check_sim:
        print("Environment successfully loaded. Press Ctrl-C in terminal to exit ...")
        while True:
            env.step_simulation(1)
    else: 
        # Try to pick up object one by one
        for obj_id in env.object_ids:  
            cnt += 1
            # Step 1: Get top down camera observation 
            rgb_obs, depth_obs, mask_gt = env.observe()
            ax1.imshow(rgb_obs)
            ax2.imshow(depth_obs)
            plt.pause(0.001)
            obj_name = env.name_list[cnt]
            input("Attempt to grasp {}. Press Enter to continue ...".format(obj_name))
            
            # Step 2: Estimate object pose 
            if args.use_state:
                # Estimating object pose with ground truth state 
                obj_pos,obj_ore = pose_est_state(obj_id, env)
            else:
                # Estimating object pose with ground truth segmentation (mask_gt)
                obj_pos,obj_ore = pose_est_segicp(obj_id, obj_name, depth_obs, mask_gt, env.camera.intrinsic_matrix, env.view_matrix)
            
            # Step 3: Compute grasp pose from object pose
            pick_pos, pick_angle = env.ob_pos_to_ee_pos(obj_pos,obj_ore,obj_id)
            
            # Step 4: Execute pick and place action
            succ = env.execute_grasp(pick_pos, pick_angle)
            if succ: # If successful place the object in target bin 
                print("grasp success")
                env.execute_place()
            else: 
                print("grasp fail")


if __name__ == '__main__':
    main()