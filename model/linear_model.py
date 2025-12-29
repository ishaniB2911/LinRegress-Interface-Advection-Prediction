"""
model.py

Linear regression model for predicting interface position in 1D advection.
Uses gradient descent optimization with MSE loss.
"""

import torch
import torch.nn as nn
import numpy as np


class LinearAdvectionModel(nn.Module):
    """
    Linear regression model: x(t) = w1*x0 + w2*u + w3*t + b
    
    In theory, for pure advection: x(t) = x0 + u*t
    So ideally: w1=1, w2=1, w3=1, b=0 (but with proper feature interaction)
    """
    
    def __init__(self, input_dim=3, output_dim=1):
        """
        Initialize linear model.
        
        Args:
            input_dim: Number of input features (x0, u, t)
            output_dim: Number of outputs (interface position)
        """
        super(LinearAdvectionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.linear(x)
    
    def get_parameters(self):
        """Return learned parameters as dictionary."""
        return {
            'weights': self.linear.weight.detach().numpy(),
            'bias': self.linear.bias.detach().numpy()
        }


class AdvectionTrainer:
    """Trainer class for the advection prediction model."""
    
    def __init__(self, model, learning_rate=0.01):
        """
        Initialize trainer.
        
        Args:
            model: LinearAdvectionModel instance
            learning_rate: Learning rate for gradient descent
        """
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, features, targets):
        """
        Train for one epoch.
        
        Args:
            features: Input features (x0, u, t)
            targets: Target interface positions
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        
        # Forward pass
        predictions = self.model(features)
        loss = self.criterion(predictions, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, features, targets):
        """
        Validate model on validation set.
        
        Args:
            features: Validation features
            targets: Validation targets
            
        Returns:
            Validation loss (MSE)
        """
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(features)
            loss = self.criterion(predictions, targets)
        
        return loss.item()
    
    def train(self, train_features, train_targets, val_features=None, 
              val_targets=None, epochs=1000, verbose=True):
        """
        Train the model for multiple epochs.
        
        Args:
            train_features: Training input features
            train_targets: Training target values
            val_features: Validation features (optional)
            val_targets: Validation targets (optional)
            epochs: Number of training epochs
            verbose: Print progress
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_features, train_targets)
            self.train_losses.append(train_loss)
            
            if val_features is not None and val_targets is not None:
                val_loss = self.validate(val_features, val_targets)
                self.val_losses.append(val_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                if val_features is not None:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
    
    def predict(self, features):
        """Make predictions on new data."""
        self.model.eval()
        with torch.no_grad():
            return self.model(features)