import numpy as np
import pybullet as p
import csv
import os
import math
import torch
import multiprocessing as mp
from typing import List, Tuple, Dict

# Import the existing environment and sampler classes
from env import BinStackEnvironment
from sampler import Sampler

class GPUParallelSampler:
    def __init__(self, 
                 num_boxes: int, 
                 width: float,
                 resolution: float, 
                 initial_box_position: List[float],
                 num_episodes: int, 
                 perfect_ratio: float, 
                 random_initial: bool,
                 num_gpus: int = None):
        """
        Initialize parallel sampler with GPU support.
        
        Args:
            num_gpus: Number of GPUs to use. If None, use all available GPUs.
        """
        # Detect available GPUs
        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        
        # Sampling parameters
        self.num_boxes = num_boxes
        self.width = width
        self.resolution = resolution
        self.initial_box_position = initial_box_position
        self.num_episodes = num_episodes
        self.perfect_ratio = perfect_ratio
        self.random_initial = random_initial
        
        # Output file management
        self.base_output_file = f'gpu_parallel_noisy_init_fixed_{num_episodes}_p{perfect_ratio}'
        
    def _run_single_gpu_sampling(self, gpu_id: int, episodes_per_gpu: int):
        """
        Run sampling on a specific GPU.
        
        Args:
            gpu_id: CUDA device ID
            episodes_per_gpu: Number of episodes to run on this GPU
        """
        # Set the specific GPU for this process
        torch.cuda.set_device(gpu_id)
        
        # Create environment with specific GPU
        env = BinStackEnvironment(gui=False)  # No GUI for parallel processing
        
        # Create sampler for this GPU with modified output file
        output_file = f'{self.base_output_file}_gpu{gpu_id}.csv'
        sampler = Sampler(
            env,
            num_boxes=self.num_boxes,
            width=self.width,
            resolution=self.resolution,
            initial_box_position=self.initial_box_position,
            num_episodes=episodes_per_gpu,
            perfect_ratio=self.perfect_ratio,
            random_initial=self.random_initial
        )
        
        # Override output file name to distinguish between GPUs
        sampler.output_file = output_file
        
        # Generate samples
        sampler.sample_and_record()
        
        # Clean up
        sampler.close()
        
    def run_parallel_sampling(self):
        """
        Run sampling across multiple GPUs in parallel.
        """
        # Check GPU availability
        if self.num_gpus == 0:
            raise RuntimeError("No CUDA-capable GPUs found.")
        
        print(f"Using {self.num_gpus} GPUs for parallel sampling")
        
        # Distribute episodes across GPUs
        episodes_per_gpu = self.num_episodes // self.num_gpus
        remainder = self.num_episodes % self.num_gpus
        
        # Prepare processes
        processes = []
        for gpu_id in range(self.num_gpus):
            # Adjust episodes to handle remainder
            current_episodes = episodes_per_gpu + (1 if gpu_id < remainder else 0)
            
            # Create process for each GPU
            p = mp.Process(
                target=self._run_single_gpu_sampling, 
                args=(gpu_id, current_episodes)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Combine CSV files
        self._combine_csv_files()
    
    def _combine_csv_files(self):
        """
        Combine CSV files from different GPUs into a single output file.
        """
        # Get all GPU output files
        gpu_files = [
            f'{self.base_output_file}_gpu{i}.csv' 
            for i in range(self.num_gpus)
        ]
        
        # Read headers from the first file
        with open(gpu_files[0], 'r') as f:
            headers = f.readline().strip()
        
        # Combine files
        final_output_file = f'{self.base_output_file}_combined.csv'
        with open(final_output_file, 'w', newline='') as outfile:
            outfile.write(headers + '\n')  # Write headers
            
            # Append data from each GPU file
            for file in gpu_files:
                with open(file, 'r') as infile:
                    next(infile)  # Skip headers
                    for line in infile:
                        outfile.write(line)
        
        print(f"Combined CSV saved to {final_output_file}")
        
        # Optional: Clean up individual GPU files
        for file in gpu_files:
            os.remove(file)

def main():
    # Create GPU-enabled parallel sampler
    gpu_sampler = GPUParallelSampler(
        num_boxes=3,  # Total number of boxes to stack
        width=1.0,
        resolution=0.01,
        initial_box_position=[0.5, 0.5, 0.],  # Fixed position for first box
        num_episodes=100,  # Total episodes across all GPUs
        perfect_ratio=0.,
        random_initial=False,
        num_gpus=None  # Use all available GPUs
    )

    # Run parallel sampling
    gpu_sampler.run_parallel_sampling()

if __name__ == "__main__":
    # Ensure multiprocessing uses spawn method for GPU compatibility
    mp.set_start_method('spawn')
    main()