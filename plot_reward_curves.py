#!/usr/bin/env python3
"""
Visualization script for comparing different reward functions.
Run this from the project root directory to avoid import conflicts.
"""

import sys
import os
import math  # Import standard library math FIRST before adding custom paths

# Now add examples/score_function to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'examples', 'score_function'))

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Import reward functions from agiqa3k module
from agiqa3k import grade_answer_l1, grade_answer_l2, grade_answer_laplace, grade_answer_gaussian


def plot_reward_functions():
    """Plot comparison of different reward functions: L1, L2, Laplace, and Gaussian."""
    
    # Set up the ground truth value
    gt = 3.0
    
    # Generate prediction values from 1.0 to 5.0
    pred_values = np.linspace(1.0, 5.0, 200)
    
    # Parameters for all functions
    r_min = 0.05
    diff_at_rmin = 1.0
    use_floor = True
    
    # Calculate rewards for each function
    rewards_l1 = [grade_answer_l1(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    rewards_l2 = [grade_answer_l2(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    rewards_laplace = [grade_answer_laplace(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    rewards_gaussian = [grade_answer_gaussian(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot each reward function
    plt.plot(pred_values, rewards_l1, label='L1 (Linear)', linewidth=2, linestyle='-', color='#1f77b4')
    plt.plot(pred_values, rewards_l2, label='L2 (Quadratic)', linewidth=2, linestyle='-', color='#ff7f0e')
    plt.plot(pred_values, rewards_laplace, label='Laplace', linewidth=2, linestyle='-', color='#2ca02c')
    plt.plot(pred_values, rewards_gaussian, label='Gaussian', linewidth=2, linestyle='-', color='#d62728')
    
    # Add ground truth line
    plt.axvline(x=gt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Ground Truth (GT={gt})')
    
    # Add horizontal line at r_min
    plt.axhline(y=r_min, color='black', linestyle=':', linewidth=1, alpha=0.5, label=f'r_min={r_min}')
    
    # Mark the diff_at_rmin points
    diff_points = [gt - diff_at_rmin, gt + diff_at_rmin]
    for x in diff_points:
        if 1.0 <= x <= 5.0:
            plt.axvline(x=x, color='purple', linestyle=':', linewidth=1, alpha=0.4)
    
    # Labels and title
    plt.xlabel('Prediction Value', fontsize=13)
    plt.ylabel('Reward', fontsize=13)
    plt.title(f'Comparison of Reward Functions (GT={gt}, r_min={r_min}, diff_at_rmin={diff_at_rmin})', 
              fontsize=14, fontweight='bold')
    
    # Grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Set axis limits
    plt.xlim(1.0, 5.0)
    plt.ylim(0, 1.05)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    output_path = 'reward_functions_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def plot_reward_functions_subplots():
    """Plot each reward function in separate subplots for detailed comparison."""
    
    # Set up the ground truth value
    gt = 3.0
    
    # Generate prediction values from 1.0 to 5.0
    pred_values = np.linspace(1.0, 5.0, 200)
    
    # Parameters for all functions
    r_min = 0.05
    diff_at_rmin = 1.0
    use_floor = True
    
    # Calculate rewards for each function
    rewards_l1 = [grade_answer_l1(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    rewards_l2 = [grade_answer_l2(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    rewards_laplace = [grade_answer_laplace(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    rewards_gaussian = [grade_answer_gaussian(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Reward Functions Comparison (GT={gt}, r_min={r_min}, diff_at_rmin={diff_at_rmin})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Plot L1
    ax1 = axes[0, 0]
    ax1.plot(pred_values, rewards_l1, linewidth=2.5, color='#1f77b4')
    ax1.axvline(x=gt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=r_min, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_title('L1 (Linear)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Prediction Value', fontsize=11)
    ax1.set_ylabel('Reward', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(1.0, 5.0)
    ax1.set_ylim(0, 1.05)
    
    # Plot L2
    ax2 = axes[0, 1]
    ax2.plot(pred_values, rewards_l2, linewidth=2.5, color='#ff7f0e')
    ax2.axvline(x=gt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=r_min, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_title('L2 (Quadratic)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Prediction Value', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(1.0, 5.0)
    ax2.set_ylim(0, 1.05)
    
    # Plot Laplace
    ax3 = axes[1, 0]
    ax3.plot(pred_values, rewards_laplace, linewidth=2.5, color='#2ca02c')
    ax3.axvline(x=gt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=r_min, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax3.set_title('Laplace', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Prediction Value', fontsize=11)
    ax3.set_ylabel('Reward', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(1.0, 5.0)
    ax3.set_ylim(0, 1.05)
    
    # Plot Gaussian
    ax4 = axes[1, 1]
    ax4.plot(pred_values, rewards_gaussian, linewidth=2.5, color='#d62728')
    ax4.axvline(x=gt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axhline(y=r_min, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax4.set_title('Gaussian', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Prediction Value', fontsize=11)
    ax4.set_ylabel('Reward', fontsize=11)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(1.0, 5.0)
    ax4.set_ylim(0, 1.05)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    output_path = 'reward_functions_subplots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


def plot_reward_functions_gradient():
    """Plot the gradient (derivative) of each reward function."""
    
    # Set up the ground truth value
    gt = 3.0
    
    # Generate prediction values from 1.0 to 5.0
    pred_values = np.linspace(1.0, 5.0, 200)
    
    # Parameters for all functions
    r_min = 0.05
    diff_at_rmin = 1.0
    use_floor = True
    
    # Calculate rewards for each function
    rewards_l1 = np.array([grade_answer_l1(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values])
    rewards_l2 = np.array([grade_answer_l2(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values])
    rewards_laplace = np.array([grade_answer_laplace(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values])
    rewards_gaussian = np.array([grade_answer_gaussian(pred, gt, r_min, diff_at_rmin, use_floor) for pred in pred_values])
    
    # Calculate gradients (numerical differentiation)
    grad_l1 = np.gradient(rewards_l1, pred_values)
    grad_l2 = np.gradient(rewards_l2, pred_values)
    grad_laplace = np.gradient(rewards_laplace, pred_values)
    grad_gaussian = np.gradient(rewards_gaussian, pred_values)
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot each gradient
    plt.plot(pred_values, grad_l1, label='L1 Gradient', linewidth=2, linestyle='-', color='#1f77b4')
    plt.plot(pred_values, grad_l2, label='L2 Gradient', linewidth=2, linestyle='-', color='#ff7f0e')
    plt.plot(pred_values, grad_laplace, label='Laplace Gradient', linewidth=2, linestyle='-', color='#2ca02c')
    plt.plot(pred_values, grad_gaussian, label='Gaussian Gradient', linewidth=2, linestyle='-', color='#d62728')
    
    # Add ground truth line
    plt.axvline(x=gt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Ground Truth (GT={gt})')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    # Labels and title
    plt.xlabel('Prediction Value', fontsize=13)
    plt.ylabel('Gradient (dReward/dPrediction)', fontsize=13)
    plt.title(f'Gradient Comparison of Reward Functions (GT={gt}, r_min={r_min}, diff_at_rmin={diff_at_rmin})', 
              fontsize=14, fontweight='bold')
    
    # Grid and legend
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # Set axis limits
    plt.xlim(1.0, 5.0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure
    output_path = 'reward_functions_gradients.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Generating Reward Function Comparison Plots")
    print("=" * 60)
    
    print("\n1. Generating overlay comparison plot...")
    try:
        plot_reward_functions()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n2. Generating subplot comparison plot...")
    try:
        plot_reward_functions_subplots()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n3. Generating gradient comparison plot...")
    try:
        plot_reward_functions_gradient()
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print("=" * 60)

