import numpy as np
import pybullet as p
import csv
import os
import math
import multiprocessing as mp
from typing import List, Tuple, Dict

from env import BinStackEnvironment
from sampler import Sampler

class ParallelSampler:
    def __init__(self, 
                 num_boxes: int, 
                 width: float,
                 resolution: float, 
                 initial_box_position: List[float],
                 num_episodes: int, 
                 perfect_ratio: float, 
                 random_initial: bool,
                 num_processes: int = None):
        """
        Initialize the parallel sampler for robot arm stacking actions.
        
        Args:
            num_boxes: Number of boxes to stack
            width: Width of the sampling cube in meters
            resolution: Resolution of the sampling grid in meters
            initial_box_position: Initial position of the first box
            num_episodes: Total number of episodes to generate
            perfect_ratio: Ratio of perfect (non-random) episodes
            random_initial: Whether to randomize initial box position
            num_processes: Number of parallel processes (defaults to CPU count)
        """
        self.num_boxes = num_boxes
        self.width = width
        self.resolution = resolution
        self.initial_box_position = initial_box_position
        self.num_episodes = num_episodes
        self.perfect_ratio = perfect_ratio
        self.random_initial = random_initial
        self.data_folder = "train_data"
        
        # Determine number of processes
        self.num_processes = num_processes or mp.cpu_count()
        
        # Calculate episodes per process
        self.episodes_per_process = [
            self.num_episodes // self.num_processes + 
            (1 if i < self.num_episodes % self.num_processes else 0) 
            for i in range(self.num_processes)
        ]
        
        # Output file naming
        self.base_output_file =  os.path.join(self.data_folder, f'train_{self.num_episodes}_p{perfect_ratio}')
        
    def _worker_process(self, 
                        process_id: int, 
                        episodes: int, 
                        queue: mp.Queue,
                        lock: mp.Lock):
        """
        Worker function to run simulation for a subset of episodes.
        
        Args:
            process_id: Unique identifier for the process
            episodes: Number of episodes to run
            queue: Multiprocessing queue for collecting results
            lock: Lock for thread-safe file writing
        """
        # Create a separate environment for this process
        env = BinStackEnvironment(gui=False)
        
        # Create a sampler for this specific process
        sampler = Sampler(
            env,
            num_boxes=self.num_boxes,
            width=self.width,
            resolution=self.resolution,
            initial_box_position=self.initial_box_position,
            num_episodes=episodes,
            perfect_ratio=self.perfect_ratio,
            random_initial=self.random_initial
        )
        
        # Create a unique output file for this process
        output_file = f'{self.base_output_file}_p{process_id}.csv'
        sampler.output_file = output_file
        
        # Initialize CSV for this process
        sampler._initialize_csv()
        
        try:
            # Run sampling for assigned episodes
            sampler.sample_and_record()
            
            # Signal completion and send output file path
            queue.put(output_file)
        except Exception as e:
            # In case of error, put the error in the queue
            queue.put(f"Error in process {process_id}: {str(e)}")
        finally:
            # Ensure environment is closed
            env.close()
            sampler.close()
    
    def merge_csv_files(self, csv_files: List[str]):
        """
        Merge multiple CSV files into a single output file.
        
        Args:
            csv_files: List of CSV file paths to merge
        """
        # Final output file
        final_output_file = f'{self.base_output_file}.csv'
        
        # Read headers from the first file
        with open(csv_files[0], 'r') as f:
            headers = f.readline().strip()
        
        # Merge files
        with open(final_output_file, 'w') as outfile:
            outfile.write(headers + '\n')  # Write headers
            
            for csv_file in csv_files:
                with open(csv_file, 'r') as infile:
                    # Skip header of subsequent files
                    next(infile)
                    # Write data
                    outfile.writelines(infile)
        
        # Optional: Clean up individual files
        for csv_file in csv_files:
            os.remove(csv_file)
        
        print(f"Merged CSV files into {final_output_file}")
    
    def sample(self):
        """
        Run parallel sampling across multiple processes.
        """
        # Create a queue and lock for inter-process communication
        result_queue = mp.Queue()
        file_lock = mp.Lock()
        
        # Create and start processes
        processes = []
        for i in range(self.num_processes):
            p = mp.Process(
                target=self._worker_process, 
                args=(i, self.episodes_per_process[i], result_queue, file_lock)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        csv_files = []
        for _ in range(self.num_processes):
            result = result_queue.get()
            if result.startswith('Error'):
                print(result)
            else:
                csv_files.append(result)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Merge CSV files
        self.merge_csv_files(csv_files)

def main():
    # Initialize parallel sampler
    parallel_sampler = ParallelSampler(
        num_boxes=3,  # Total number of boxes to stack (including initial box)
        width=1.0,
        resolution=0.01,
        initial_box_position=[0.5, 0.5, 0.],  # Fixed position for first box
        num_episodes=50000,  # Total episodes
        perfect_ratio=0.,
        random_initial=True,
        num_processes=None  # Use all available CPU cores
    )
    
    # Generate samples in parallel
    parallel_sampler.sample()

if __name__ == "__main__":
    # Ensure multiprocessing uses 'spawn' method for compatibility
    mp.set_start_method('spawn')
    main()