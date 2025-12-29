"""
Configuration file for 1D advection interface prediction
"""

# Physical parameters
VELOCITY = 1.0  # Advection velocity (m/s)
DOMAIN_LENGTH = 10.0  # Domain length (m)
INITIAL_POSITION = 2.0  # Initial interface position (m)

# Data generation parameters
TIME_START = 0.0
TIME_END = 5.0
N_SAMPLES = 100  # Number of time samples for training

# Test parameters
N_TEST_SAMPLES = 30

# Model parameters
LEARNING_RATE = 0.01
N_EPOCHS = 1000
BATCH_SIZE = 32

# Random seed for reproducibility
RANDOM_SEED = 42
