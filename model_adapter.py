# Model Adapter for UAV Simulation
# Handles compatibility between different model input/output formats

import numpy as np
from utils import CONFIG


# Expected model input dimensions
MODEL_INPUT_DIM = 250

def pad_observation(observation, target_dim=MODEL_INPUT_DIM):
    """
    Pad observation vector to match the expected model input dimension.
    
    Args:
        observation: Original observation vector from UAV.get_observation()
        target_dim: Target dimension for the model input
        
    Returns:
        Padded observation vector with correct dimensions
    """
    # Get the actual observation dimension
    actual_dim = observation.shape[0]
    
    # Check if padding is needed
    if actual_dim >= target_dim:
        # If observation is already larger or equal, return the first target_dim elements
        return observation[:target_dim]
    
    # Create the padded observation
    padded = np.zeros(target_dim, dtype=np.float32)
    
    # Copy the original observation to the beginning of the padded array
    padded[:actual_dim] = observation
    
    return padded

def create_compatible_input(uav, model_input_dim=MODEL_INPUT_DIM):
    """
    Create a compatible input for the model from UAV state.
    
    This function constructs a model input that matches the expected dimensions
    by combining the UAV's observation with additional data and padding.
    
    Args:
        uav: UAV object or observation vector
        model_input_dim: Expected model input dimension
        
    Returns:
        numpy.ndarray: Model input with correct dimensions
    """
    # Check if input is already an observation vector
    if isinstance(uav, np.ndarray):
        observation = uav
    # Otherwise, get observation from UAV object
    else:
        # If the object has a get_observation method, use it
        if hasattr(uav, 'get_observation'):
            try:
                observation = uav.get_observation()
            except Exception as e:
                print(f"Error getting observation: {e}")
                # Fallback to a basic observation vector
                observation = np.zeros(28, dtype=np.float32)  # Default observation size
        else:
            # Handle case where uav isn't a proper UAV object
            print("Warning: Input is not a proper UAV object with get_observation method")
            observation = np.zeros(28, dtype=np.float32)  # Default observation size
    
    # Pad to match expected dimensions
    return pad_observation(observation, model_input_dim)

def adapt_model_output(model_output, action_dim=2):
    """
    Adapt model output to expected action format.
    
    Args:
        model_output: Output from the neural network model
        action_dim: Expected action dimension
        
    Returns:
        numpy.ndarray: Adapted action vector
    """
    # Check if model_output is a scalar (float)
    if isinstance(model_output, (np.float32, np.float64, float)):
        # If it's a scalar, create a default action vector
        return np.array([model_output, 0.0], dtype=np.float32)
    
    # If it's an array, ensure it has the right dimensions
    model_output_array = np.asarray(model_output)
    
    # Ensure model output has the right number of dimensions for our action space
    if model_output_array.size >= action_dim:
        return model_output_array[:action_dim]
    else:
        # If model output is too small, pad with zeros
        padded = np.zeros(action_dim, dtype=np.float32)
        padded[:model_output_array.size] = model_output_array.flatten()
        return padded
