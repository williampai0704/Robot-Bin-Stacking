import torch
import numpy as np
import pybullet as p
from env import BinStackEnvironment
from RLmodel import DQN
from typing import List, Tuple, Dict
import math
import csv

class PolicyActionSelector:
    def __init__(self, env, model_path, state_dim, n_actions, resolution, num_boxes, num_inference):
        """
        Initialize the policy-based action selector.
        
        Args:
            model_path (str): Path to the saved PyTorch model
            state_dim (int): Dimension of the state space
            n_actions (int): Total number of discretized actions
            resolution (float): Resolution of action space
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Recreate the DQN model architecture
        self.policy_net = DQN(state_dim, n_actions).to(self.device)
        
        # Load the trained model
        model = torch.load('target_net.pt', map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(model)
        self.policy_net.eval()  # Set to evaluation mode
        
        self.env = env
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.resolution = resolution
        self.num_boxes = num_boxes
        self.num_inference = num_inference
        self.output_file = f'inference_{self.num_inference}.csv'
        
    def ind2coord(self, action_index, low=0., high=1.):
        """
        Convert action index to continuous coordinates.
        
        Args:
            action_index (int): Discretized action index
            low (float): Lower bound of action space
            high (float): Upper bound of action space
        
        Returns:
            numpy.ndarray: Continuous (x, z) coordinates
        """
        width = high - low
        num = int(width/self.resolution)
        coord = np.unravel_index(action_index, (num+1, num+1))
        ax = (coord[0] / num) * width + low
        az = (coord[1] / num) * width + low
        return np.array([ax, az])
    
    
    def select_action(self, state):
        """
        Select the best action based on the current state using the trained policy.
        
        Args:
            state (torch.Tensor): Current state representation
            initial_box_position (list): Initial position of the first box
        
        Returns:
            tuple: Selected (x, z) coordinates for action
        """
        with torch.no_grad():
            # Ensure state is a tensor and on the correct device
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get Q-values for all actions
            q_values = self.policy_net(state)
            
            # Select the action with the highest Q-value
            action_index = q_values.argmax().item()
            print(action_index)
            
            # Convert action index to continuous coordinates
            action_coords = self.ind2coord(action_index)
            
            # Adjust coordinates relative to initial box position
            x =  action_coords[0]
            z =  action_coords[1]
            
            return x, z
    
    def _get_state_info(self, box_id: int) -> Dict:
        """Get position and dimensions of a box."""
        position, orientation = p.getBasePositionAndOrientation(box_id)
        dimensions = self.env.boxes[box_id]
        return {
            'position': position,
            'dimensions': dimensions
        }
    def _initialize_csv(self):
        """Initialize CSV file with appropriate headers based on number of boxes."""
        headers = []
        
        # Headers for initial box 
        headers.extend(['box_0_x', 'box_0_z'])
        headers.extend(['box_0_l', 'box_0_h'])
        
        # Headers for current box being placed
        headers.extend(['box_c_l', 'box_c_h'])
        
        # Headers for previously placed boxes (excluding initial box)
        for i in range(self.num_boxes - 2):
            prefix = f'box_{i+1}'
            headers.extend([
                f'{prefix}_x', f'{prefix}_z',
                f'{prefix}_l', f'{prefix}_h'
            ])
        
        # Action and reward headers
        headers.extend([
            'a_x', 'a_z',  # Action (end effector position)
            'reward',  # Reward
            'efficiency', 'collision_penalty', 'stack_penalty'  # Additional metrics
        ])
        
        with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    def prepare_state(self, placed_boxes):
        """
        Prepare the state representation for the policy network.
        
        Args:
            initial_box_dim (list): Dimensions of the initial box
            current_box_dim (list): Dimensions of the current box
            placed_boxes (list): List of previously placed boxes
        
        Returns:
            torch.Tensor: Prepared state vector
        """
        # Initialize state vector
        state = []
        print("Placed boxes:", placed_boxes)
        # Initial box position and dimensions
        initial_box_id = placed_boxes[0]
        initial_box_state = self._get_state_info(initial_box_id)
        initial_box_pos = _to_3_decimals(initial_box_state["position"])
        initial_box_dim = _to_3_decimals(initial_box_state["dimensions"])
        state.extend([
            initial_box_pos[0],  # x
            initial_box_pos[2],  # z
            initial_box_dim[0],  # length
            initial_box_dim[2]   # height
        ])
        current_box_id = placed_boxes[-1]
        current_box_state = self._get_state_info(current_box_id)
        current_box_dim = current_box_state["dimensions"]
        # Current box dimensions
        state.extend([current_box_dim[0], current_box_dim[2]])
        prev_placed_boxes = placed_boxes[:-1]
        
        print("Previous placed boxes:", prev_placed_boxes)
        # Add previously placed boxes' information
        for i in range(self.num_boxes - 2):
            if i < len(prev_placed_boxes) - 1:
                box_id = prev_placed_boxes[i+1]
                box_state = self._get_state_info(box_id)
                box_pos = box_state["position"]
                box_dim = box_state["dimensions"]
                state.extend([_to_3_decimals(box_pos[0]),_to_3_decimals(box_pos[2])]) 
                state.extend([_to_3_decimals(box_dim[0]),_to_3_decimals(box_dim[1])])
            else:
                # Padding for boxes not yet placed
                state.extend([-1.0,-1.0,0.0,0.0])  # 2 for position, 2 for dimensions
        print("box_0_x,box_0_z,box_0_l,box_0_h,box_c_l,box_c_h,box_1_x,box_1_z,box_1_l,box_1_h")
        print(state)
        return state
    
    def record_inference(self, state, action, reward_info):
        state.extend([
            action[0], action[1],
            reward_info['total_reward'],
            reward_info['efficiency_ratio'],
            reward_info['collision_penalty'],
            reward_info['stack_penalty'],
        ])
        print("box_0_x,box_0_z,box_0_l,box_0_h,box_c_l,box_c_h,box_1_x,box_1_z,box_1_l,box_1_h,a_x,a_z,reward,efficiency,collision_penalty,stack_penalty")
        print(state)
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(state)
            
  
def _to_3_decimals(num):
        if isinstance(num, (int, float)):
            return math.trunc(num * 1000) / 1000
        elif isinstance(num, list):
            return [math.trunc(x * 1000) / 1000 for x in num]
        elif isinstance(num, tuple):
            return tuple(math.trunc(x * 1000) / 1000 for x in num)
        else:
            raise TypeError("Input must be an int, float, or list of numbers")     
        
def apply_policy_in_simulation(policy_selector):
    """
    Apply the trained policy to stack boxes in the simulation.
    
    Args:
        env (BinStackEnvironment): Simulation environment
        policy_selector (PolicyActionSelector): Trained policy selector
        initial_box_position (list): Initial position of the first box
        num_boxes (int): Total number of boxes to stack
    """
    num_boxes = policy_selector.num_boxes
    num_inference = policy_selector.num_inference
    initial_sample_x = [0.3, 0.5]
    
    for infernce in range(num_inference):
        # Reset environment and set up initial box
        placed_boxes = []
        initial_box_position = [_to_3_decimals(np.random.uniform(initial_sample_x[0], initial_sample_x[1])),
                            0.5,
                            0.0]
        initial_box_id, initial_box_dim = policy_selector.env.load_box()
        p.resetBasePositionAndOrientation(
            initial_box_id, 
            initial_box_position,
            p.getQuaternionFromEuler([0, 0, 0])
        )
        policy_selector.env.step_simulation(100)  # Let physics settle
        placed_boxes.append(initial_box_id)
        
        # Stack remaining boxes
        for _ in range(num_boxes - 1):
            # Load new box to be placed
            current_box_id, current_box_dim = policy_selector.env.load_box()
            p.resetBasePositionAndOrientation(current_box_id, [0.5, -0.5, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
            placed_boxes.append(current_box_id)
            # Prepare state representation
            state = policy_selector.prepare_state(placed_boxes)
            
            # Select action using trained policy
            action_x, action_z = policy_selector.select_action(state)
            
            # Execute action
            grasp_success = policy_selector.env.execute_grasp(current_box_id, current_box_dim)
            
            if grasp_success:
                # Move to selected position
                policy_selector.env.move_tool(
                    position=[action_x, initial_box_position[1], action_z],
                    orientation=p.getQuaternionFromEuler([np.pi, 0, 0])
                )
                
                # Release box and let it settle
                policy_selector.env.release_box()
                policy_selector.env.step_simulation(500)
                
                reward_info = policy_selector.env.get_total_reward()
                # Add current box to placed boxes
                
                
                policy_selector.record_inference(
                            state=state, 
                            action=(action_x, action_z),
                            reward_info=reward_info
                        )
            # Return to home position
            policy_selector.env.robot_go_home()
        
        # Clear all boxes except the initial one for next episode
        for box_id in placed_boxes:
            p.removeBody(box_id)  
        policy_selector.env.boxes.clear() 
        print(f"Completed inferdnce {num_inference + 1}/{num_inference}")  

# Example usage
def main():
    # Initialize environment
    env = BinStackEnvironment(gui=True)
    
    # Initialize policy selector
    policy_selector = PolicyActionSelector(
        env = env,
        model_path="model.pt",  # Path to your saved model
        state_dim=10,  # As defined in your training script
        n_actions=11**2,  # As defined in your training script
        resolution=0.1,  # As defined in your training script
        num_boxes=3,
        num_inference = 5
    )
    policy_selector._initialize_csv()
    # Apply policy in simulation
    
   
    apply_policy_in_simulation(policy_selector)
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()