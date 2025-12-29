"""
data_generator.py

Generates synthetic data for 1D advection interface tracking problem.
The interface position evolves according to: x(t) = x0 + u*t
where x0 is initial position and u is advection velocity.
"""

import numpy as np
import torch


class AdvectionDataGenerator:
    """Generate training/test data for 1D advection interface prediction."""
    
    def __init__(self, velocity_range=(0.5, 2.0), x0_range=(0.0, 5.0), 
                 t_max=10.0, noise_level=0.0):
        """
        Initialize data generator.
        
        Args:
            velocity_range: (min, max) advection velocities
            x0_range: (min, max) initial positions
            t_max: Maximum time for simulation
            noise_level: Gaussian noise standard deviation
        """
        self.velocity_range = velocity_range
        self.x0_range = x0_range
        self.t_max = t_max
        self.noise_level = noise_level
    
    def generate_dataset(self, n_samples, n_time_steps=100):
        """
        Generate dataset with multiple advection cases.
        
        Args:
            n_samples: Number of different advection cases
            n_time_steps: Time steps per case
            
        Returns:
            features: (n_samples * n_time_steps, 3) - [x0, u, t]
            targets: (n_samples * n_time_steps, 1) - interface position x(t)
        """
        features_list = []
        targets_list = []
        
        for _ in range(n_samples):
            # Random initial conditions
            x0 = np.random.uniform(*self.x0_range)
            u = np.random.uniform(*self.velocity_range)
            
            # Time array
            t = np.linspace(0, self.t_max, n_time_steps)
            
            # Interface position: x(t) = x0 + u*t
            x_t = x0 + u * t
            
            # Add noise if specified
            if self.noise_level > 0:
                x_t += np.random.normal(0, self.noise_level, x_t.shape)
            
            # Create feature matrix [x0, u, t]
            features = np.column_stack([
                np.full(n_time_steps, x0),
                np.full(n_time_steps, u),
                t
            ])
            
            features_list.append(features)
            targets_list.append(x_t.reshape(-1, 1))
        
        # Concatenate all samples
        features = np.vstack(features_list)
        targets = np.vstack(targets_list)
        
        return torch.FloatTensor(features), torch.FloatTensor(targets)
    
    def generate_test_case(self, x0, u, n_time_steps=100):
        """Generate a single test case with specific parameters."""
        t = np.linspace(0, self.t_max, n_time_steps)
        x_t = x0 + u * t
        
        features = np.column_stack([
            np.full(n_time_steps, x0),
            np.full(n_time_steps, u),
            t
        ])
        
        return torch.FloatTensor(features), torch.FloatTensor(x_t.reshape(-1, 1))