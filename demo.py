import numpy as np
import pandas as pd
import pybullet as p
import math

from env import BinStackEnvironment

def run_top_reward_simulations(csv_file, num_top_simulations=10, gui=True):
    """
    Run simulations for the top reward rollouts from the CSV file.
    
    Args:
        csv_file (str): Path to the CSV file with rollout data
        num_top_simulations (int): Number of top reward simulations to run
        gui (bool): Whether to run in GUI mode
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Sort by total reward and get top indices
    idx = np.argsort(df['reward'])[::-1][:num_top_simulations]
    
    # Function to truncate to 3 decimal places
    def to_3_decimals(num):
        return math.trunc(num * 1000) / 1000
    
    # Prepare output report
    report = []
    
    # Run simulation for each top rollout
    env = BinStackEnvironment(gui=gui)
    for i, index in enumerate(idx, 1):
        
        try:
            # Extract data for this rollout
            rollout_data = df.loc[index]
            
            # Set initial box
            initial_box_id, _ = env.load_box(length=rollout_data['box_0_l'], height=rollout_data['box_0_h'])
            p.resetBasePositionAndOrientation(
                initial_box_id, 
                [to_3_decimals(rollout_data['box_0_x']), 0.5, to_3_decimals(rollout_data['box_0_z'])],
                p.getQuaternionFromEuler([0, 0, 0])
            )
    
            env.step_simulation(100)  # Let physics settle
            
            # Tracking placed boxes
            placed_boxes = [initial_box_id]
            
            # Place second box
            second_box_id, second_box_dim = env.load_box(length=rollout_data['box_c_l'], height=rollout_data['box_c_h'])
            p.resetBasePositionAndOrientation(
                second_box_id, 
                [0.5, -0.5, 0.1],  # Default initial position
                p.getQuaternionFromEuler([0, 0, 0])
            )
            
            # Execute grasp for second box
            grasp_success = env.execute_grasp(second_box_id, second_box_dim)
            
            if grasp_success:
                # Move to sampled action position for second box
                env.move_tool(
                    position=[to_3_decimals(rollout_data['a_x']), 0.5, to_3_decimals(rollout_data['a_z'])],
                    orientation=p.getQuaternionFromEuler([np.pi, 0, 0])
                )
                
                # Release box and let it settle
                env.release_box()
                env.step_simulation(500)
                
                placed_boxes.append(second_box_id)
                
                # Place third box
                # First, ensure we're in the next row of the DataFrame
                next_index = index + 1
                if next_index < len(df):
                    third_box_id, third_box_dim = env.load_box(length=df.loc[next_index, 'box_c_l'], height=df.loc[next_index, 'box_c_h'])
                    p.resetBasePositionAndOrientation(
                        third_box_id, 
                        [0.5, -0.5, 0.1],  # Default initial position
                        p.getQuaternionFromEuler([0, 0, 0])
                    )
                    
                    # Execute grasp for third box
                    grasp_success = env.execute_grasp(third_box_id, third_box_dim)
                    
                    if grasp_success:
                        # Move to sampled action position for third box
                        env.move_tool(
                            position=[to_3_decimals(df.loc[next_index, 'a_x']), 0.5, to_3_decimals(df.loc[next_index, 'a_z'])],
                            orientation=p.getQuaternionFromEuler([np.pi, 0, 0])
                        )
                        
                        # Release box and let it settle
                        env.release_box()
                        env.step_simulation(500)
                        
                        placed_boxes.append(third_box_id)
                        
                        # Get final reward
                        final_reward = env.get_total_reward()
                        
                        # Collect report information
                        report.append({
                            'simulation_num': i,
                            'original_index': index,
                            'original_reward': rollout_data['reward'],
                            'final_reward': final_reward['total_reward'],
                            'efficiency_ratio': final_reward['efficiency_ratio'],
                            'collision_penalty': final_reward['collision_penalty'],
                            'stack_penalty': final_reward['stack_penalty']
                        })
            
            # Return to home position
            env.robot_go_home()
            
            # Clean up environment
            for box_id in placed_boxes:
                p.removeBody(box_id)
        except Exception as e:
            print(f"Error in simulation {i} with index {index}: {e}")

    env.close()
        

    
    # Convert report to DataFrame for easy viewing
    return pd.DataFrame(report)

def main():
    results = run_top_reward_simulations("inference_101.csv", num_top_simulations=10, gui=True)
    print("\nTop Reward Simulation Results:")
    print(results)

if __name__ == "__main__":
    main()