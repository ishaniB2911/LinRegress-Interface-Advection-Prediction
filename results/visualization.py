"""
Visualization utilities for 1D advection interface prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_training_data(t_train: np.ndarray, x_train: np.ndarray):
    """
    Plot training data
    
    Args:
        t_train: Training time values
        x_train: Training position values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(t_train, x_train, alpha=0.6, label='Training data')
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Interface Position (m)', fontsize=12)
    plt.title('Training Data: 1D Advection Interface', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions(t_train: np.ndarray, x_train: np.ndarray,
                    t_test: np.ndarray, x_test: np.ndarray,
                    x_pred: np.ndarray):
    """
    Plot predictions vs actual values
    
    Args:
        t_train: Training time values
        x_train: Training position values
        t_test: Test time values
        x_test: Test position values
        x_pred: Predicted position values
    """
    plt.figure(figsize=(12, 6))
    
    plt.scatter(t_train, x_train, alpha=0.5, s=50, 
                label='Training data', color='blue')
    plt.scatter(t_test, x_test, alpha=0.7, s=50, 
                label='Test data (actual)', color='green', marker='s')
    plt.plot(t_test, x_pred, 'r-', linewidth=2, 
             label='Predictions', alpha=0.8)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Interface Position (m)', fontsize=12)
    plt.title('Model Predictions vs Actual Interface Position', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_cost_history(cost_history: List[float]):
    """
    Plot cost function during training
    
    Args:
        cost_history: List of cost values
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cost (MSE)', fontsize=12)
    plt.title('Training Cost History', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()


def plot_residuals(t_test: np.ndarray, x_test: np.ndarray, 
                   x_pred: np.ndarray):
    """
    Plot prediction residuals
    
    Args:
        t_test: Test time values
        x_test: Test position values
        x_pred: Predicted position values
    """
    residuals = x_test - x_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs time
    axes[0].scatter(t_test, residuals, alpha=0.6, s=50)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Time (s)', fontsize=12)
    axes[0].set_ylabel('Residual (m)', fontsize=12)
    axes[0].set_title('Residuals vs Time', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    
    # Residual histogram
    axes[1].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Residual (m)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Residual Distribution', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def create_full_report(t_train: np.ndarray, x_train: np.ndarray,
                       t_test: np.ndarray, x_test: np.ndarray,
                       x_pred: np.ndarray, cost_history: List[float],
                       train_mse: float, test_mse: float,
                       true_velocity: float, learned_velocity: float,
                       true_position: float, learned_position: float):
    """
    Create comprehensive visualization report
    
    Args:
        t_train: Training time values
        x_train: Training position values
        t_test: Test time values
        x_test: Test position values
        x_pred: Predicted position values
        cost_history: Training cost history
        train_mse: Training MSE
        test_mse: Test MSE
        true_velocity: True advection velocity
        learned_velocity: Learned velocity
        true_position: True initial position
        learned_position: Learned initial position
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Training data
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(t_train, x_train, alpha=0.6, s=40, color='blue')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('Training Data')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cost history
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(cost_history, linewidth=2, color='purple')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Cost (MSE)')
    ax2.set_title('Training Cost History')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. Predictions
    ax3 = fig.add_subplot(gs[1, :])
    ax3.scatter(t_train, x_train, alpha=0.4, s=40, label='Training', color='blue')
    ax3.scatter(t_test, x_test, alpha=0.6, s=50, label='Test (actual)', 
                color='green', marker='s')
    ax3.plot(t_test, x_pred, 'r-', linewidth=2.5, label='Predictions', alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Model Predictions vs Actual')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuals
    residuals = x_test - x_pred
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(t_test, residuals, alpha=0.6, s=50, color='orange')
    ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Residual (m)')
    ax4.set_title('Prediction Residuals')
    ax4.grid(True, alpha=0.3)
    
    # 5. Statistics text box
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    stats_text = f"""
    MODEL PERFORMANCE METRICS
    ═══════════════════════════════════
    
    Training MSE:     {train_mse:.6f}
    Test MSE:         {test_mse:.6f}
    
    RMSE (Test):      {np.sqrt(test_mse):.6f}
    Max Error:        {np.max(np.abs(residuals)):.6f}
    Mean Error:       {np.mean(residuals):.6f}
    
    LEARNED PARAMETERS
    ═══════════════════════════════════
    
    True Velocity:        {true_velocity:.4f} m/s
    Learned Velocity:     {learned_velocity:.4f} m/s
    Velocity Error:       {abs(true_velocity - learned_velocity):.4f}
    
    True Initial Pos:     {true_position:.4f} m
    Learned Initial Pos:  {learned_position:.4f} m
    Position Error:       {abs(true_position - learned_position):.4f}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.3))
    
    plt.suptitle('1D Advection Interface Prediction - Full Report', 
                 fontsize=16, fontweight='bold')
    plt.show()
