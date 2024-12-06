import numpy as np
import pybullet as p
import csv
import os
import math
import torch
import multiprocessing as mp
from typing import List, Tuple, Dict
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('gpu_sampling.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import the existing environment and sampler classes
# Assumes these are in the same directory or in PYTHONPATH
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
            num_boxes: Number of boxes to stack
            width: Width of the bin/environment
            resolution: Sampling resolution
            initial_box_position: Starting position of first box
            num_episodes: Total number of sampling episodes
            perfect_ratio: Ratio of perfect placements
            random_initial: Whether to randomize initial placement
            num_gpus: Number of GPUs to use. If None, use all available GPUs.
        """
        # Detect and validate available GPUs
        try:
            self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
            if self.num_gpus == 0:
                logging.warning("No CUDA-capable GPUs found. Falling back to CPU.")
                self.num_gpus = 1
        except Exception as e:
            logging.error(f"GPU detection failed: {e}")
            self.num_gpus = 1
        
        # Sampling parameters
        self.num_boxes = num_boxes
        self.width = width
        self.resolution = resolution
        self.initial_box_position = initial_box_position
        self.num_episodes = num_episodes
        self.perfect_ratio = perfect_ratio
        self.random_initial = random_initial
        
        # Output file management
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_file = f'binstack_sampling_{timestamp}'
        
        logging.info(f"Initialized GPUParallelSampler with {self.num_gpus} GPU(s)")
    
    def _validate_cuda(self):
        """
        Validate CUDA environment and capabilities.
        """
        if not torch.cuda.is_available():
            logging.warning("CUDA not available. Falling back to CPU sampling.")
            return False
        
        gpu_name = torch.cuda.get_device_name(0)
        logging.info(f"Primary GPU: {gpu_name}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        
        return True
    
    def _run_single_gpu_sampling(self, gpu_id: int, episodes_per_gpu: int):
        """
        Run sampling on a specific GPU.
        
        Args:
            gpu_id: CUDA device ID
            episodes_per_gpu: Number of episodes to run on this GPU
        """
        try:
            # Set the specific GPU for this process
            torch.cuda.set_device(gpu_id)
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
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
            
            logging.info(f"GPU {gpu_id} sampling completed: {episodes_per_gpu} episodes")
        
        except Exception as e:
            logging.error(f"Error in GPU {gpu_id} sampling: {e}")
            raise
    
    def run_parallel_sampling(self):
        """
        Run sampling across multiple GPUs in parallel.
        """
        # Validate CUDA environment
        if not self._validate_cuda():
            logging.warning("Falling back to single-process sampling")
            # Implement fallback single-process sampling here if needed
            return
        
        # Distribute episodes across GPUs
        episodes_per_gpu = self.num_episodes // self.num_gpus
        remainder = self.num_episodes % self.num_gpus
        
        # Prepare processes
        processes = []
        for gpu_id in range(self.num_gpus):
            # Adjust episodes to handle remainder
            current_episodes = episodes_per_gpu + (1 if gpu_id < remainder else 0)
            
            # Create process for each GPU
            process = mp.Process(
                target=self._run_single_gpu_sampling, 
                args=(gpu_id, current_episodes)
            )
            processes.append(process)
            process.start()
        
        # Wait for all processes to complete
        try:
            for process in processes:
                process.join()
            
            # Combine CSV files after successful completion
            self._combine_csv_files()
        
        except Exception as e:
            logging.error(f"Parallel sampling failed: {e}")
            # Implement error recovery or fallback strategy
    
    def _combine_csv_files(self):
        """
        Combine CSV files from different GPUs into a single output file.
        """
        try:
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
            
            logging.info(f"Combined CSV saved to {final_output_file}")
            
            # Optional: Clean up individual GPU files
            for file in gpu_files:
                os.remove(file)
        
        except Exception as e:
            logging.error(f"CSV combination failed: {e}")

def main():
    """
    Main execution function for GPU-accelerated bin stacking sampling.
    """
    try:
        # Create GPU-enabled parallel sampler
        gpu_sampler = GPUParallelSampler(
            num_boxes=3,                  # Total number of boxes to stack
            width=1.0,                    # Bin width
            resolution=0.01,              # Sampling resolution
            initial_box_position=[0.5, 0.5, 0.],  # Fixed position for first box
            num_episodes=10,             # Total episodes across all GPUs
            perfect_ratio=0.4,             # Proportion of perfect placements
            random_initial=True,
            num_gpus=None                 # Automatically detect GPUs
        )

        # Run parallel sampling
        gpu_sampler.run_parallel_sampling()
    
    except Exception as e:
        logging.error(f"Sampling process failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Required for GPU multiprocessing
    from datetime import datetime
    mp.set_start_method('spawn', force=True)
    
    # Run main function
    main()