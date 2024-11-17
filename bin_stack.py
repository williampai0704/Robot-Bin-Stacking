import numpy as np
import time
import argparse
import time
import pybullet as p
from matplotlib import pyplot as plt
from env import BinStackEnviornment

def main():
    # Initialize environment
    env = BinStackEnviornment(gui=True)
    time.sleep(1)  # Give time for GUI to initialize
    
    # Define stacking location
    stack_location = [0.5, 0.3, 0.]  # Location where boxes will be stacked
    
    # Generate three boxes at different positions
    box_positions = [
        [0.5, -0.3, 0.1],  # First box
        [0.2, -0.3, 0.1],  # Second box
        [0.8, -0.3, 0.1]   # Third box
    ]
    
    boxes = []
    for pos in box_positions:
        box_id, box_dim = env.load_box()
        p.resetBasePositionAndOrientation(box_id, pos, p.getQuaternionFromEuler([0, 0, 0]))
        boxes.append((box_id, box_dim))
        time.sleep(0.5)  # Give physics time to settle
    
    # Stack boxes one by one
    current_height = 0
    for i, (box_id, box_dim) in enumerate(boxes):
        print(f"Stacking box {i+1}")
        
        # Move to home position
        env.robot_go_home()
        time.sleep(0.5)
        
        # Calculate placement position (each box stacks on top of previous)
        place_pos = np.array(stack_location) + np.array([0, 0, current_height + box_dim[2] + 0.01])
        current_height += box_dim[2] # Update height for next box
        
        # Execute picking sequence
        print(f"Picking box {i+1}")
        success = env.execute_grasp(box_id, box_dim)
        if not success:
            print(f"Failed to grasp box {i+1}")
            continue
            
        time.sleep(0.5)  # Give time for grasp to settle
        
        # Move to placement position
        print(f"Placing box {i+1}")
        env.move_tool(
            position=place_pos,
            orientation=p.getQuaternionFromEuler([np.pi, 0, 0])
        )
        time.sleep(0.5)  # Wait for movement to complete
        
        # Release box
        env.release_box()
        time.sleep(0.5)  # Let physics settle
        
        # Move up slightly
        env.move_tool(
            position=place_pos + np.array([0, 0, 0.1]),
            orientation=p.getQuaternionFromEuler([0, np.pi, 0])
        )
        
        print(f"Box {i+1} stacked successfully")
    
    # Return to home position
    env.robot_go_home()
    
    # Calculate final stacking efficiency
    efficiency = env.calculate_stacking_efficiency()
    print(f"\nFinal stacking efficiency: {efficiency:.2%}")
    
    # Keep the simulation running
    while p.isConnected():
        env.step_simulation(1)
        time.sleep(1./240.)

if __name__ == "__main__":
    main()