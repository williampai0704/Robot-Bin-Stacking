# Robot-Bin-Stacking

## Project Overview
This project implements a robotic bin stacking simulation using PyBullet.

## Environment Setup

### Prerequisites
- Anaconda or Miniconda installed
- Python 3.9+

### Create Conda Environment
1. Clone the repository
```bash
git clone <your-repository-url>
cd robot-bin-stacking
```

2. Create the conda environment using the provided environment.yaml
```bash
conda env create -f environment.yaml
```

3. Activate the environment
```bash
conda activate robot-bin-stacking
```

## Sampling Configuration

### Parallel Sampling Parameters
In `mp_sampler.py`, you can adjust the following key parameters:

- `num_episodes`: Total number of episodes to generate (default: 1000)
  - Increase for more training data
  - Decrease for faster testing

- `perfect_ratio`: Ratio of perfect (non-random) episodes (range: 0.0 to 1.0, default: 0.2)
  - 1.0: All episodes use optimal/deterministic actions
  - 0.0: All episodes use random actions
  - Intermediate values create a mix of strategies

### Example Configuration
```python
parallel_sampler = ParallelSampler(
    num_boxes=3,         # Total number of boxes to stack
    width=1.0,           # Sampling cube width
    resolution=0.01,     # Grid resolution
    initial_box_position=[0.5, 0.5, 0.],  # Initial box placement
    num_episodes=500,    # Number of episodes (adjust as needed)
    perfect_ratio=0.8,   # 80% perfect, 20% random episodes
    random_initial=True,
    num_processes=None   # Use all available CPU cores
)
```

## Running the Sampling Script
```bash
python3 mp_sampler.py
```

## Output
- Generated data will be saved in the `train_data` folder
- CSV filename format: `train_{num_episodes}_p{perfect_ratio}.csv`
  - Example: `train_500_p0.8.csv`

### CSV Columns
- `box_0_x`: Initial box x coordinate
- `box_0_z`: Initial box z coordinate
- `box_0_l`: Initial box length
- `box_0_h`: Initial box height
- `box_c_l`: Current box length
- `box_c_h`: Current box height
- `box_1_x`: box 1 x coordinate (-1.0 if hasn't been placed)
- `box_1_z`: box 1 z coordinate (-1.0 if hasn't been placed)
- `box_1_l`: box 1 length (0.0 if hasn't been placed)
- `box_1_h`: box 1 height (0.0 if hasn't been placed)
- `a_x`: Action x
- `a_z`: Action z
- `reward`: Episode reward
- `efficiency`: Stacking efficiency ratio
- `collision_penalty`: Collision-related penalty
- `stack_penalty`: Stacking-related penalty