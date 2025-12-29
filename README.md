# 1D Advection Interface Position Prediction

## Overview
This project implements a linear regression model using gradient descent to predict interface position in a 1D advection problem. The model is built with PyTorch and demonstrates applied CFD/flow modeling using machine learning.

## Physical Problem
In 1D advection, an interface moves with constant velocity according to:
```
x(t) = x₀ + u·t
```
where:
- `x(t)` is the interface position at time t
- `x₀` is the initial position
- `u` is the advection velocity
- `t` is time

## Project Structure
```
.
├──data
├────data_generator.py      # Generates synthetic advection data
├──model
├────linear_model.py        # Linear regression model and trainer
├──results
├────predictions.png
├────training_history.png   # Visulisation files
├── main.py                 # Main execution script
└── README.md               # This file
```

## Files Description

### `data_generator.py`
- **Class**: `AdvectionDataGenerator`
- Generates synthetic training and test data
- Creates multiple advection scenarios with varying initial conditions
- Supports adding Gaussian noise for realism
- Output: Feature matrix `[x₀, u, t]` and target `x(t)`

### `linear_model.py`
- **Class**: `LinearAdvectionModel` - Linear regression model
  - Architecture: `x(t) = w₁·x₀ + w₂·u + w₃·t + b`
  - Ideally learns: `w₁≈1, w₂≈1, w₃≈1, b≈0` (physical consistency)
  
- **Class**: `AdvectionTrainer` - Training manager
  - Loss function: Mean Squared Error (MSE)
  - Optimizer: Stochastic Gradient Descent (SGD)
  - Tracks training and validation losses

### `main.py`
Main execution script that:
1. Generates training/validation datasets
2. Initializes and trains the model
3. Evaluates performance using MSE, RMSE, MAE, R²
4. Visualizes training history and predictions
5. Tests on specific cases

## Installation

### Requirements
```bash
pip install torch numpy matplotlib
```

Or create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch numpy matplotlib
```

## Usage

### Basic Execution
```bash
python3 main.py
```

### Expected Output
1. **Console Output**:
   - Training progress (every 100 epochs)
   - Final MSE on training and validation sets
   - Learned model parameters
   - Evaluation metrics (MSE, RMSE, MAE, R²)

2. **Visualizations**:
   - `training_history.png` - Loss curves over epochs
   - `predictions.png` - Predicted vs true interface positions

### Example Output
```
Training set size: 5000 samples
Validation set size: 1000 samples

Epoch 100/1000 - Train Loss: 0.023456, Val Loss: 0.024123
Epoch 200/1000 - Train Loss: 0.012345, Val Loss: 0.013012
...

Final Training MSE: 0.002543
Final Validation MSE: 0.002678

Learned Parameters:
Weights (w1, w2, w3): [0.998, 0.995, 1.002]
Bias (b): [0.003]

Validation Metrics:
  MSE: 0.002678
  RMSE: 0.051749
  MAE: 0.041234
  R2: 0.999876
```

## Model Performance

The linear regression model effectively learns the advection physics:
- **MSE** measures average squared prediction error
- Model weights should approximate the physical relationship
- R² score near 1.0 indicates excellent fit
- Small noise is handled through regularization by SGD

## Customization

### Modify Data Generation
```python
data_gen = AdvectionDataGenerator(
    velocity_range=(0.5, 2.0),  # Range of velocities
    x0_range=(0.0, 5.0),        # Range of initial positions
    t_max=10.0,                  # Maximum simulation time
    noise_level=0.05             # Gaussian noise std dev
)
```

### Adjust Training Parameters
```python
trainer = AdvectionTrainer(
    model, 
    learning_rate=0.01  # Adjust learning rate
)

trainer.train(
    train_features, train_targets,
    epochs=1000,  # Number of epochs
    verbose=True
)
```

## Extension Ideas

1. **Non-linear advection**: Add velocity-dependent terms
2. **2D advection**: Extend to 2D interface tracking
3. **Neural networks**: Replace linear model with MLP
4. **Real CFD data**: Train on OpenFOAM or other CFD simulations
5. **Physics-informed loss**: Add PDE residual to loss function

## References
- 1D Advection equation: ∂u/∂t + c·∂u/∂x = 0
- PyTorch documentation: https://pytorch.org/docs/
- CFD applications of ML: Various research papers on data-driven fluid dynamics

## License
MIT License - Feel free to use and modify for your projects.
