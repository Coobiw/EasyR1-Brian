# Reward Function Visualization

This document explains how to visualize and compare the different reward functions (L1, L2, Laplace, Gaussian) implemented in `examples/score_function/agiqa3k.py`.

## Files

- **`plot_reward_curves.py`**: Main visualization script (located in project root)
- **`examples/score_function/plot_reward_functions.py`**: Alternative script (in score_function directory)

## Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install matplotlib numpy
```

Or install all project requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run from project root (Recommended)

```bash
cd /path/to/EasyR1-Brian
python plot_reward_curves.py
```

### Option 2: Using a specific Python environment

If you have environment issues, activate the correct environment first:

```bash
# If using conda
conda activate your_env_name
python plot_reward_curves.py

# If using virtualenv
source your_venv/bin/activate
python plot_reward_curves.py
```

## Output Files

The script generates three PNG files in the project root directory:

1. **`reward_functions_comparison.png`**: Overlay plot showing all four reward functions on the same axes
2. **`reward_functions_subplots.png`**: 2×2 subplot grid showing each function separately
3. **`reward_functions_gradients.png`**: Comparison of the gradients (derivatives) of each function

## Reward Functions Explained

All functions are configured with:
- **Ground Truth (GT)**: 3.0
- **r_min**: 0.05 (minimum reward floor)
- **diff_at_rmin**: 1.0 (difference at which reward reaches r_min)
- **Prediction range**: [1.0, 5.0]

### L1 (Linear)
- **Formula**: `reward = 1 - coeff * |pred - gt|`
- **Coefficient**: `coeff = (1 - r_min) / diff_at_rmin`
- **Characteristic**: Linear decay, uniform penalty for errors

### L2 (Quadratic)
- **Formula**: `reward = 1 - coeff * (pred - gt)²`
- **Coefficient**: `coeff = (1 - r_min) / (diff_at_rmin)²`
- **Characteristic**: Quadratic decay, gentle for small errors, severe for large errors

### Laplace
- **Formula**: `reward = exp(-d / τ)`
- **Parameter**: `τ = d0 / log(1 / r_min)` where `d` is normalized difference
- **Characteristic**: Exponential decay with heavier tails than Gaussian

### Gaussian
- **Formula**: `reward = exp(-(d² / (2σ²)))`
- **Parameter**: `σ = d0 / sqrt(2 * log(1 / r_min))` where `d` is normalized difference
- **Characteristic**: Bell-shaped curve, smooth decay

## Troubleshooting

### ImportError: No module named 'matplotlib'

```bash
pip install matplotlib
```

### ImportError: No module named 'numpy'

```bash
pip install numpy
```

### NumPy version compatibility issues

If you see errors about NumPy 1.x vs 2.x:

```bash
# Option 1: Downgrade numpy
pip install "numpy<2"

# Option 2: Upgrade matplotlib
pip install --upgrade matplotlib
```

### Module 'math' conflict

If you get errors about `math.py` importing from `mathruler`:
- Make sure you run the script from the **project root directory**
- The script is designed to avoid this conflict by importing standard library `math` before adding custom paths

## Customization

You can modify the parameters in the script to visualize different scenarios:

```python
# In plot_reward_curves.py, modify these values:
gt = 3.0              # Ground truth value
r_min = 0.05          # Minimum reward floor
diff_at_rmin = 1.0    # Difference threshold
use_floor = True      # Whether to use reward floor
```

## Example Output Description

### Overlay Comparison Plot
Shows all four reward functions on the same plot for direct comparison. The vertical dashed line indicates the ground truth value, and the horizontal dotted line shows the r_min floor.

### Subplot Comparison
Each reward function is shown in its own subplot, making it easier to see individual characteristics without visual clutter.

### Gradient Comparison
Shows how the rate of change (gradient) differs between functions. Useful for understanding how each function will affect gradient-based optimization during training.

## Implementation Details

The reward functions are implemented in `examples/score_function/agiqa3k.py`:

- **Low-level functions**: `grade_answer_l1()`, `grade_answer_l2()`, etc.
- **Wrapper functions**: `accuracy_reward_l1()`, `accuracy_reward_l2()`, etc.
- **Complete scoring**: `compute_score_l1()`, `compute_score_l2()`, etc.

Each function includes:
- Range validation (predictions must be in [1.0, 5.0])
- Configurable r_min floor
- Configurable diff_at_rmin threshold
- Optional floor clamping

