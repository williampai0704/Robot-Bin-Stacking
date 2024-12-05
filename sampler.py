import numpy as np
import pybullet as p
import csv
from typing import List, Tuple, Dict
import os
import math
from env import BinStackEnviornment

class Sampler:
    def __init__(self, env, num_boxes: int, width: float,
                 resolution: float, initial_box_position: List[float],
                 num_episodes: int, perfect_ratio: float):
        """
        Initialize the sampler for robot arm stacking actions.
        
        Args:
            env: The BinStackEnvironment instance
            width: Width of the sampling cube in meters
            resolution: Resolution of the sampling grid in meters
            output_file: Path to save the sampling data
        """
        self.env = env
        self.num_boxes = num_boxes
        self.width = width  # dimension of the action space 
        self.resolution = resolution
        self.initial_box_position = initial_box_position
        self.num_episodes = num_episodes
        self.perfect_ratio = perfect_ratio

        # self.output_file = 'stacking_samples_random.csv_'+ str(self.num_episodes)
        self.output_file = f'stacking_samples_mixed_{self.num_episodes}_p{perfect_ratio}.csv'

        # self.num_samples = int(width / resolution)
        num_perfect_episodes = int(num_episodes * perfect_ratio)
        self.num_random_episodes = num_episodes - num_perfect_episodes
        
        # Initialize CSV file with headers
        self._initialize_csv()
    
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

    def _get_state_info(self, box_id: int) -> Dict:
        """Get position and dimensions of a box."""
        position, orientation = p.getBasePositionAndOrientation(box_id)
        dimensions = self.env.boxes[box_id]
        return {
            'position': position,
            'dimensions': dimensions
        }
        
    def sample_random_action(self) -> Tuple[float, float]:
        """
        Generate a random action within the sampling cube.
        
        Args:
            base_position: Base position of the current box
            
        Returns:
            Tuple of (x, z) coordinates for end effector position
        """
        # Generate random offsets within the sampling cube
        x_offset = np.random.uniform(-self.width/2, self.width/2)
        z_offset = np.random.uniform(0, self.width)  # Only positive z offsets
        
        # Calculate absolute positions
        x = self.initial_box_position[0] + x_offset
        z = self.initial_box_position[2] + z_offset
        
        # Round to resolution
        x = self.truncate_to_3_decimals(round(x / self.resolution) * self.resolution)
        z = self.truncate_to_3_decimals(round(z / self.resolution) * self.resolution)
        
        return x, z
    
    def sample_perfect_action(self, placed_boxes: List[Tuple[int, List[float]]], current_box_dim: List[float]) -> Tuple[float, float]:
        """
        Generate a perfect stacking action based on previous box positions.
        
        Args:
            placed_boxes: List of previously placed boxes
            
        Returns:
            Tuple of (x, z) coordinates for end effector position
        """
        if not placed_boxes:
            # If no boxes placed yet, use initial position
            return self.initial_box_position[0], self.initial_box_position[2]
        
        # Get the last placed box's position and dimensions
        last_box_id, last_box_dim = placed_boxes[-1]
        last_box_pos, _ = p.getBasePositionAndOrientation(last_box_id)
        
        # Perfect placement strategy: 
        # 1. Align x-coordinate with the previous box 
        # 2. Place z-coordinate slightly above the previous box
        # NOTE: the action coordinate is the tip of the robotic arm, not the center of the box
        x = last_box_pos[0]
        z = last_box_pos[2] + last_box_dim[2]/2 + current_box_dim[2] + 0.05  # Small offset
        
        return self.truncate_to_3_decimals(x), self.truncate_to_3_decimals(z)
    
    def setup_initial_box(self) -> Tuple[int, List[float]]:
        """Set up the initial fixed position box."""
        box_id, box_dim = self.env.load_box()
        p.resetBasePositionAndOrientation(
            box_id, 
            self.initial_box_position,
            p.getQuaternionFromEuler([0, 0, 0])
        )
        self.env.step_simulation(100)  # Let physics settle
        return box_id, box_dim
    
    def truncate_to_3_decimals(self, num):
        return math.trunc(num * 1000) / 1000

    def _record_episode_step(self, initial_box_dim: List[float], 
                           current_box_dim: List[float],
                           placed_boxes: List[Tuple[int, List[float]]],
                           action: Tuple[float, float],
                           reward_info: Dict):
        """Record a single step to the CSV file."""
        row = []
        
        # Add initial box dimensions
        initial_box_id, _ = placed_boxes[0]
        initial_box_pos,_ = p.getBasePositionAndOrientation(initial_box_id)
        row.extend([self.truncate_to_3_decimals(initial_box_pos[0]),self.truncate_to_3_decimals(initial_box_pos[2])])
        row.extend([initial_box_dim[0],initial_box_dim[2]])
        
        # Add current box dimensions
        row.extend([current_box_dim[0],current_box_dim[2]])
        
        # Add previously placed boxes' information
        for i in range(self.num_boxes - 2):
            if i < len(placed_boxes) - 1:
                box_id, box_dim = placed_boxes[i+1]
                pos, _ = p.getBasePositionAndOrientation(box_id)
                row.extend([self.truncate_to_3_decimals(pos[0]),self.truncate_to_3_decimals(pos[2])]) 
                row.extend([self.truncate_to_3_decimals(box_dim[0]),self.truncate_to_3_decimals(box_dim[1])])
            else:
                # Padding for boxes not yet placed
                row.extend([-1.0,-1.0,0.0,0.0])  # 2 for position, 2 for dimensions
        row.extend([
            action[0], action[1],
            reward_info['total_reward'],
            reward_info['efficiency_ratio'],
            reward_info['collision_penalty'],
            reward_info['stack_penalty'],
        ])
        
        with open(self.output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def sample_and_record(self,):
        """
        Generate samples and record the results for multiple episodes.
        
        Args:
            num_episodes: Number of complete stacking episodes to simulate
        """
        for episode in range(self.num_episodes):
            # Reset environment and set up initial box
            placed_boxes = []
            initial_box_id, initial_box_dim = self.setup_initial_box()
            placed_boxes.append((initial_box_id, initial_box_dim))
            
            # Stack remaining boxes
            for _ in range(self.num_boxes - 1):
                # Load new box to be placed
                current_box_id, current_box_dim = self.env.load_box()
                
                p.resetBasePositionAndOrientation(current_box_id, [0.5,-0.5,0.1], p.getQuaternionFromEuler([0, 0, 0]))
                if episode < self.num_random_episodes:
                    # Generate random action
                    action_x, action_z = self.sample_random_action()
                else:
                    # Generate perfect action
                    action_x, action_z = self.sample_perfect_action(placed_boxes, current_box_dim)
                
                # Execute action
                grasp_success = self.env.execute_grasp(current_box_id, current_box_dim)
                
                if grasp_success:
                    # Move to sampled position
                    self.env.move_tool(
                        position=[action_x, self.initial_box_position[1], action_z],
                        orientation=p.getQuaternionFromEuler([np.pi, 0, 0])
                    )
                    
                    # Release box and let it settle
                    self.env.release_box()
                    self.env.step_simulation(500)
                    
                    # Get reward and record sample
                    reward_info = self.env.get_total_reward()
                    placed_boxes.append((current_box_id, current_box_dim))
                    
                    # Record sample
                    self._record_episode_step(
                        initial_box_dim=initial_box_dim,
                        current_box_dim=current_box_dim,
                        placed_boxes=placed_boxes[:-1],  # Exclude current box
                        action=(action_x, action_z),
                        reward_info=reward_info
                    )
                
                # Return to home position
                self.env.robot_go_home()
            
            # Clear all boxes except the initial one for next episode
            for box_id, _ in placed_boxes:
                p.removeBody(box_id)
                self.env.boxes = {}
                
            print(f"Completed episode {episode + 1}/{self.num_episodes}")
                
    def close(self):
        """Clean up resources."""
        if self.env:
            p.disconnect()
            
            
            
def main():
    # Initialize environment and sampler
    env = BinStackEnviornment(gui=True)  # Set gui=False for faster sampling
    sampler = Sampler(
        env,
        num_boxes=3,  # Total number of boxes to stack (including initial box)
        width=1.0,
        resolution=0.1,
        initial_box_position=[0.5, 0.5, 0.],  # Fixed position for first box
        num_episodes=10,
        perfect_ratio=1.
    )

    # Generate samples
    sampler.sample_and_record()

    # Clean up
    sampler.close()

if __name__ == "__main__":
    main()
    
