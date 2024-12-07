import torch
import numpy as np
import pybullet as p
from env import BinStackEnviornment
from RLmodel import DQN
import math

class PolicyActionSelector:
    def __init__(self, model_path, state_dim=10, n_actions=11**2, resolution=0.1, num_boxes = 3):
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
        self.policy_net.load_state_dict(torch.load(model_path))
        self.policy_net.eval()  # Set to evaluation mode
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.resolution = resolution
        self.num_boxes = num_boxes
        
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
    
    def truncate_to_3_decimals(self, num):
        return math.trunc(num * 1000) / 1000
    
    def select_action(self, state, initial_box_position):
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
    
    def prepare_state(self, initial_box_dim, current_box_dim, placed_boxes):
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
        
        # Initial box position and dimensions
        initial_box_id, _ = placed_boxes[0]
        initial_box_pos, _ = p.getBasePositionAndOrientation(initial_box_id)
        state.extend([
            initial_box_pos[0],  # x
            initial_box_pos[2],  # z
            initial_box_dim[0],  # length
            initial_box_dim[2]   # height
        ])
        
        # Current box dimensions
        state.extend([
            current_box_dim[0],  # length
            current_box_dim[2]   # height
        ])
        
        # Add previously placed boxes' information
        for i in range(self.num_boxes - 2):
            if i < len(placed_boxes) - 1:
                box_id, box_dim = placed_boxes[i+1]
                pos, _ = p.getBasePositionAndOrientation(box_id)
                state.extend([self.truncate_to_3_decimals(pos[0]),self.truncate_to_3_decimals(pos[2])]) 
                state.extend([self.truncate_to_3_decimals(box_dim[0]),self.truncate_to_3_decimals(box_dim[1])])
            else:
                # Padding for boxes not yet placed
                state.extend([-1.0,-1.0,0.0,0.0])  # 2 for position, 2 for dimensions
        
        return state

def apply_policy_in_simulation(env, policy_selector, initial_box_position, num_boxes):
    """
    Apply the trained policy to stack boxes in the simulation.
    
    Args:
        env (BinStackEnviornment): Simulation environment
        policy_selector (PolicyActionSelector): Trained policy selector
        initial_box_position (list): Initial position of the first box
        num_boxes (int): Total number of boxes to stack
    """
    # Reset environment and set up initial box
    placed_boxes = []
    initial_box_id, initial_box_dim = env.load_box()
    p.resetBasePositionAndOrientation(
        initial_box_id, 
        initial_box_position,
        p.getQuaternionFromEuler([0, 0, 0])
    )
    env.step_simulation(100)  # Let physics settle
    placed_boxes.append((initial_box_id, initial_box_dim))
    
    # Stack remaining boxes
    for _ in range(num_boxes - 1):
        # Load new box to be placed
        current_box_id, current_box_dim = env.load_box()
        p.resetBasePositionAndOrientation(current_box_id, [0.5, -0.5, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
        
        # Prepare state representation
        state = policy_selector.prepare_state(initial_box_dim, current_box_dim, placed_boxes)
        print(state)
        
        # Select action using trained policy
        action_x, action_z = policy_selector.select_action(state, initial_box_position)
        print(action_x, action_z)
        
        # Execute action
        grasp_success = env.execute_grasp(current_box_id, current_box_dim)
        
        if grasp_success:
            # Move to selected position
            env.move_tool(
                position=[action_x, initial_box_position[1], action_z],
                orientation=p.getQuaternionFromEuler([np.pi, 0, 0])
            )
            
            # Release box and let it settle
            env.release_box()
            env.step_simulation(500)
            reward_info = env.get_total_reward()
            print(reward_info)
            
            # Add current box to placed boxes
            placed_boxes.append((current_box_id, current_box_dim))
        
        # Return to home position
        env.robot_go_home()
    
    return placed_boxes

# Example usage
def main():
    # Initialize environment
    env = BinStackEnviornment(gui=False)
    
    # Initialize policy selector
    policy_selector = PolicyActionSelector(
        model_path="/home/irmak/project-2/final-project/policy_net.pt",  # Path to your saved model
        state_dim=10,  # As defined in your training script
        n_actions=101**2,  # As defined in your training script
        resolution=0.01,  # As defined in your training script
        num_boxes=3
    )
    
    # Apply policy in simulation
    xpos = np.random.uniform(0.3, 0.5, 50)
    for x in xpos:
        x = np.round(x, 3)
        initial_box_position = [x, 0.5, 0.]
        apply_policy_in_simulation(env, policy_selector, initial_box_position, num_boxes=3)


    # initial_box_position = [0.5, 0.5, 0.]
    # apply_policy_in_simulation(env, policy_selector, initial_box_position, num_boxes=3)
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()