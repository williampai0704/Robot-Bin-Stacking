import numpy as np
import pybullet as p
import torch
import multiprocessing
import csv
from typing import List, Tuple, Dict
import os
import math
from env import BinStackEnvironment
import concurrent.futures

class CUDASampler:
    def __init__(self, 
                 num_boxes: int, 
                 width: float,
                 resolution: float, 
                 initial_box_position: List[float],
                 num_episodes: int, 
                 perfect_ratio: float, 
                 random_initial: bool,
                 num_cuda_streams: int = None):
        """
        Initialize CUDA-accelerated sampler for robot arm stacking actions.
        
        Args:
            num_boxes: Total number of boxes to stack
            width: Width of the sampling cube in meters
            resolution: Resolution of the sampling grid in meters
            initial_box_position: Initial position of the first box
            num_episodes: Total number of episodes to simulate
            perfect_ratio: Ratio of perfect placement episodes
            random_initial: Whether to randomize initial box position
            num_cuda_streams: Number of CUDA streams to use (defaults to GPU multiprocessing cores)
        """
        self.num_boxes = num_boxes
        self.width = width
        self.resolution = resolution
        self.num_episodes = num_episodes
        self.perfect_ratio = perfect_ratio
        
        # Detect and configure CUDA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_cuda_streams = num_cuda_streams or torch.cuda.device_count()
        
        # Initial box position setup
        initial_sample_x = [0.3, 0.5]
        self.initial_box_position = (
            [self._to_3_decimals(np.random.uniform(initial_sample_x[0], initial_sample_x[1])), 0.5, 0.0]
            if random_initial else initial_box_position
        )
        
        # Output file configuration
        self.output_file = f'cuda_noisy_init_fixed_{self.num_episodes}_p{perfect_ratio}.csv'
        
        # Calculate perfect and random episode splits
        self.num_perfect_episodes = int(num_episodes * perfect_ratio)
        self.num_random_episodes = num_episodes - self.num_perfect_episodes
        
        # Pre-allocate CUDA tensors for parallel computations
        self._initialize_cuda_tensors()
    
    def _initialize_cuda_tensors(self):
        """
        Pre-allocate CUDA tensors for efficient parallel computations.
        """
        # Generate random action spaces
        self.random_x_actions = torch.rand(self.num_episodes, device=self.device) * self.width - self.width/2
        self.random_z_actions = torch.rand(self.num_episodes, device=self.device) * self.width
        
        # Noise tensors for adding variations
        noise_std = 0.02  # Convert to scalar
        self.position_noise = torch.normal(
            mean=torch.zeros(self.num_episodes, 3, device=self.device), 
            std=noise_std
        )
        self.dimension_noise = torch.normal(
            mean=torch.zeros(self.num_episodes, 3, device=self.device), 
            std=noise_std
        )
    
    def _simulate_episode(self, episode_index: int):
        """
        Simulate a single stacking episode using PyBullet.
        
        Args:
            episode_index: Index of the current episode
        
        Returns:
            Dictionary of episode results
        """

        try:
            p.getConnectionInfo()
        except:
            p.connect(p.DIRECT)  # Use DIRECT mode for headless simulation
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.8)
        # Create a separate environment for each simulation to avoid conflicts
        env = BinStackEnvironment(gui=False)
        
        # Setup initial box
        placed_boxes = []
        initial_box_id, initial_box_dim = self._setup_initial_box(env)
        placed_boxes.append(initial_box_id)
        
        # Stack remaining boxes
        for _ in range(self.num_boxes - 1):
            # Load new box
            current_box_id, current_box_dim = env.load_box()
            
            # Determine action type (random or perfect)
            is_perfect = episode_index >= self.num_random_episodes
            
            if is_perfect:
                action_x, action_z = self._get_perfect_action(env, placed_boxes, current_box_dim)
            else:
                action_x, action_z = self._get_random_action()
            
            # Execute grasp and placement
            grasp_success = self._execute_placement(env, current_box_id, current_box_dim, 
                                                    action_x, action_z)
            
            if grasp_success:
                # Get reward and record sample
                reward_info = env.get_total_reward()
                placed_boxes.append(current_box_id)
                
                # Optional: Return or process episode data
                return {
                    'placed_boxes': placed_boxes,
                    'action': (action_x, action_z),
                    'reward_info': reward_info
                }
            
        return None
        
        # Clean up environment
        env.close()
    
    def sample_and_record_parallel(self):
        """
        Generate samples in parallel using CUDA and multiprocessing.
        """
        # Use concurrent futures for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cuda_streams) as executor:
            # Submit episodes to parallel execution
            futures = [
                executor.submit(self._simulate_episode, episode) 
                for episode in range(self.num_episodes)
            ]
            
            # Initialize CSV and write results
            self._initialize_csv()
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    self._record_episode_step(
                        placed_boxes=result['placed_boxes'],
                        action=result['action'],
                        reward_info=result['reward_info']
                    )
    
    def _get_random_action(self) -> Tuple[float, float]:
        """Generate a random action within the sampling cube."""
        x_offset = self.random_x_actions[0].cpu().numpy()
        z_offset = self.random_z_actions[0].cpu().numpy()
        
        x = self.initial_box_position[0] + x_offset
        z = self.initial_box_position[2] + z_offset
        
        return (
            self._to_3_decimals(round(x / self.resolution) * self.resolution),
            self._to_3_decimals(round(z / self.resolution) * self.resolution)
        )
    
    def _get_perfect_action(self, env, placed_boxes, current_box_dim):
        """Generate a perfect stacking action."""
        # Similar to original perfect action logic
        if not placed_boxes:
            return self.initial_box_position[0], self.initial_box_position[2]
        
        last_box_id = placed_boxes[-1]
        last_box_pos, _ = p.getBasePositionAndOrientation(last_box_id)
        last_box_dim = env.boxes[last_box_id]
        
        x = last_box_pos[0]
        z = last_box_pos[2] + last_box_dim[2]/2 + current_box_dim[2] + 0.05
        
        return self._to_3_decimals(x), self._to_3_decimals(z)
    
    def _execute_placement(self, env, current_box_id, current_box_dim, action_x, action_z):
        """Execute box placement in the environment."""
        grasp_success = env.execute_grasp(current_box_id, current_box_dim)
        
        if grasp_success:
            env.move_tool(
                position=[action_x, self.initial_box_position[1], action_z],
                orientation=p.getQuaternionFromEuler([np.pi, 0, 0])
            )
            
            env.release_box()
            env.step_simulation(500)
            
            return True
        return False
    
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
    def _record_episode_step(self, placed_boxes, action, reward_info):
        """Record a single step to the CSV file."""
        row = []
        
        # Add initial box dimensions
        initial_box_id = placed_boxes[0]
        initial_box_state = self._get_state_info(initial_box_id)
        initial_box_pos,initial_box_dim = self.add_noise(initial_box_state)
        row.extend([initial_box_pos[0],initial_box_pos[2]])
        row.extend([initial_box_dim[0],initial_box_dim[2]])
        
        # Add current box dimensions
        current_box_id = placed_boxes[-1]
        current_box_state = self._get_state_info(current_box_id)
        current_box_pos,current_box_dim = self.add_noise(current_box_state)
        row.extend([current_box_dim[0],current_box_dim[2]])
        
        # Add previously placed boxes' information
        prev_placed_boxes = placed_boxes[:-1]
        for i in range(self.num_boxes - 2):
            if i < len(prev_placed_boxes) - 1:
                box_id = prev_placed_boxes[i+1]
                
                box_state = self._get_state_info(box_id)
                box_pos,box_dim = self.add_noise(box_state)
                row.extend([self._to_3_decimals(box_pos[0]),self._to_3_decimals(box_pos[2])]) 
                row.extend([self._to_3_decimals(box_dim[0]),self._to_3_decimals(box_dim[2])])
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
    
    
    def _setup_initial_box(self, env):
        """Set up the initial fixed position box."""
        box_id, box_dim = env.load_box()
        p.resetBasePositionAndOrientation(
            box_id, 
            self.initial_box_position,
            p.getQuaternionFromEuler([0, 0, 0])
        )
        env.step_simulation(100)
        return box_id, box_dim
    
    def _to_3_decimals(self, num):
        """Truncate number to 3 decimal places."""
        if isinstance(num, (int, float)):
            return math.trunc(num * 1000) / 1000
        elif isinstance(num, list):
            return [math.trunc(x * 1000) / 1000 for x in num]
        else:
            raise TypeError("Input must be an int, float, or list of numbers")

def main():
    # Initialize CUDA-accelerated sampler
    cuda_sampler = CUDASampler(
        num_boxes=3,
        width=1.0,
        resolution=0.01,
        initial_box_position=[0.5, 0.5, 0.],
        num_episodes=10,
        perfect_ratio=0.4,
        random_initial=True,
        num_cuda_streams=None  # Automatically detect available CUDA cores
    )
    
    # Generate samples in parallel
    cuda_sampler.sample_and_record_parallel()

if __name__ == "__main__":
    main()