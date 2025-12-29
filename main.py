"""
main.py

Main script for training and evaluating linear regression model
for 1D advection interface position prediction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.data_generator import AdvectionDataGenerator
from model.linear_model import LinearAdvectionModel, AdvectionTrainer


def plot_training_history(trainer, save_path='training_history.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.train_losses, label='Training Loss', linewidth=2)
    if trainer.val_losses:
        plt.plot(trainer.val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_predictions(features, true_targets, predictions, 
                     case_idx=0, n_time_steps=100, save_path='predictions.png'):
    """Plot predicted vs true interface positions for a test case."""
    start_idx = case_idx * n_time_steps
    end_idx = start_idx + n_time_steps
    
    t = features[start_idx:end_idx, 2].numpy()
    true_x = true_targets[start_idx:end_idx].numpy().flatten()
    pred_x = predictions[start_idx:end_idx].numpy().flatten()
    
    x0 = features[start_idx, 0].item()
    u = features[start_idx, 1].item()
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, true_x, 'b-', label='True Position', linewidth=2)
    plt.plot(t, pred_x, 'r--', label='Predicted Position', linewidth=2)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Interface Position x(t)', fontsize=12)
    plt.title(f'Interface Position Prediction\n(x0={x0:.2f}, u={u:.2f})', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Prediction plot saved to {save_path}")


def calculate_metrics(true_targets, predictions):
    """Calculate various error metrics."""
    mse = torch.mean((predictions - true_targets) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(predictions - true_targets)).item()
    
    # R² score
    ss_res = torch.sum((true_targets - predictions) ** 2).item()
    ss_tot = torch.sum((true_targets - torch.mean(true_targets)) ** 2).item()
    r2 = 1 - (ss_res / ss_tot)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}


def main():
    """Main execution function."""
    
    print("="*60)
    print("1D ADVECTION INTERFACE POSITION PREDICTION")
    print("Linear Regression with Gradient Descent (PyTorch)")
    print("="*60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize data generator
    print("\n[1] Generating training and validation data...")
    data_gen = AdvectionDataGenerator(
        velocity_range=(0.5, 2.0),
        x0_range=(0.0, 5.0),
        t_max=10.0,
        noise_level=0.05  # Small noise for realism
    )
    
    # Generate datasets
    train_features, train_targets = data_gen.generate_dataset(
        n_samples=50, n_time_steps=100
    )
    val_features, val_targets = data_gen.generate_dataset(
        n_samples=10, n_time_steps=100
    )
    
    print(f"Training set size: {train_features.shape[0]} samples")
    print(f"Validation set size: {val_features.shape[0]} samples")
    print(f"Feature dimensions: {train_features.shape[1]} (x0, u, t)")
    
    # Initialize model
    print("\n[2] Initializing linear regression model...")
    model = LinearAdvectionModel(input_dim=3, output_dim=1)
    print(f"Model architecture: Linear(3 -> 1)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize trainer
    print("\n[3] Training model with gradient descent...")
    trainer = AdvectionTrainer(model, learning_rate=0.01)
    
    # Train the model
    trainer.train(
        train_features, train_targets,
        val_features, val_targets,
        epochs=1000,
        verbose=True
    )
    
    # Final training metrics
    print("\n[4] Training complete!")
    final_train_loss = trainer.train_losses[-1]
    final_val_loss = trainer.val_losses[-1]
    print(f"Final Training MSE: {final_train_loss:.6f}")
    print(f"Final Validation MSE: {final_val_loss:.6f}")
    
    # Get learned parameters
    params = model.get_parameters()
    print("\n[5] Learned Parameters:")
    print(f"Weights (w1, w2, w3): {params['weights'].flatten()}")
    print(f"Bias (b): {params['bias'].flatten()}")
    print("\nTheoretical model: x(t) = x0 + u*t")
    print("Expected: w1≈1 (x0 coeff), w2≈1 (u coeff), w3≈1 (t coeff), b≈0")
    
    # Test on validation set
    print("\n[6] Evaluating on validation set...")
    val_predictions = trainer.predict(val_features)
    val_metrics = calculate_metrics(val_targets, val_predictions)
    
    print("Validation Metrics:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Test on specific case
    print("\n[7] Testing on specific case (x0=2.0, u=1.5)...")
    test_features, test_targets = data_gen.generate_test_case(
        x0=2.0, u=1.5, n_time_steps=100
    )
    test_predictions = trainer.predict(test_features)
    test_metrics = calculate_metrics(test_targets, test_predictions)
    
    print("Test Case Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Generate plots
    print("\n[8] Generating visualizations...")
    plot_training_history(trainer)
    plot_predictions(val_features, val_targets, val_predictions, case_idx=0)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()