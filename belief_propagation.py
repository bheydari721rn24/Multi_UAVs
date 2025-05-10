"""
Bayesian Belief Propagation Module for Multi-UAV Systems

This module implements distributed belief updates using Bayesian inference,
allowing UAVs to maintain and share probabilistic beliefs about the environment.

Author: Research Team
Date: 2025
"""

import numpy as np
import math
from utils import CONFIG
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter

class BeliefPropagator:
    """
    Bayesian belief propagation for distributed state estimation
    Maintains and updates probabilistic beliefs about environment state
    """
    
    def __init__(self, scenario_width=CONFIG["scenario_width"], 
                 scenario_height=CONFIG["scenario_height"]):
        """Initialize belief propagator with configuration parameters"""
        self.config = CONFIG["swarm"]["belief_propagation"]
        self.belief_grid_resolution = self.config["belief_grid_resolution"]
        self.observation_noise_sigma = self.config["observation_noise_sigma"]
        self.process_noise_sigma = self.config["process_noise_sigma"]
        self.confidence_decay_rate = self.config["confidence_decay_rate"]
        self.min_update_threshold = self.config["min_belief_update_threshold"]
        
        self.scenario_width = scenario_width
        self.scenario_height = scenario_height
        
        # Initialize belief grid for target position
        # Higher values indicate higher probability of target presence
        self.target_belief = np.ones(self.belief_grid_resolution) / np.prod(self.belief_grid_resolution)
        
        # Initialize confidence in belief (0-1)
        self.belief_confidence = 0.2  # Start with low confidence
        
        # Last known target state
        self.last_target_state = None
        self.last_update_time = 0
        
        # Cell size in world coordinates
        self.cell_width = scenario_width / self.belief_grid_resolution[0]
        self.cell_height = scenario_height / self.belief_grid_resolution[1]
        
        # For obstacle beliefs - dictionary mapping obstacles to belief precision
        self.obstacle_beliefs = {}
        
    def world_to_grid(self, position):
        """
        Convert world coordinates to grid indices
        
        Args:
            position: World position [x, y]
            
        Returns:
            tuple: Grid indices (i, j)
        """
        i = min(int(position[0] / self.cell_width), self.belief_grid_resolution[0] - 1)
        j = min(int(position[1] / self.cell_height), self.belief_grid_resolution[1] - 1)
        return i, j
        
    def grid_to_world(self, grid_indices):
        """
        Convert grid indices to world coordinates
        
        Args:
            grid_indices: Grid indices (i, j)
            
        Returns:
            numpy.ndarray: World position [x, y]
        """
        x = (grid_indices[0] + 0.5) * self.cell_width
        y = (grid_indices[1] + 0.5) * self.cell_height
        return np.array([x, y])
        
    def predict_belief(self, target_velocity=None, dt=CONFIG["time_step"]):
        """
        Predict belief forward in time using target velocity if available
        
        Args:
            target_velocity: Estimated target velocity [vx, vy]
            dt: Time step for prediction
            
        Returns:
            numpy.ndarray: Updated belief grid
        """
        # If no velocity information, apply diffusion
        if target_velocity is None:
            # Apply Gaussian blur as diffusion
            sigma = self.process_noise_sigma
            self.target_belief = gaussian_filter(self.target_belief, sigma=sigma)
            
            # Decay confidence due to prediction uncertainty
            self.belief_confidence *= self.confidence_decay_rate
            
            return self.target_belief
            
        # With velocity, apply translation and diffusion
        
        # Calculate grid cell shift based on velocity
        dx = int(round(target_velocity[0] * dt / self.cell_width))
        dy = int(round(target_velocity[1] * dt / self.cell_height))
        
        # Shift belief grid
        if dx != 0 or dy != 0:
            shifted_belief = np.zeros_like(self.target_belief)
            
            # Calculate valid source and target ranges for shifting
            src_i_min = max(0, -dx)
            src_i_max = min(self.belief_grid_resolution[0], self.belief_grid_resolution[0] - dx)
            src_j_min = max(0, -dy)
            src_j_max = min(self.belief_grid_resolution[1], self.belief_grid_resolution[1] - dy)
            
            tgt_i_min = max(0, dx)
            tgt_i_max = min(self.belief_grid_resolution[0], self.belief_grid_resolution[0] + dx)
            tgt_j_min = max(0, dy)
            tgt_j_max = min(self.belief_grid_resolution[1], self.belief_grid_resolution[1] + dy)
            
            # Shift values
            shifted_belief[tgt_i_min:tgt_i_max, tgt_j_min:tgt_j_max] = \
                self.target_belief[src_i_min:src_i_max, src_j_min:src_j_max]
                
            # Apply diffusion to account for prediction uncertainty
            sigma = self.process_noise_sigma
            self.target_belief = gaussian_filter(shifted_belief, sigma=sigma)
            
            # Decay confidence due to prediction uncertainty
            self.belief_confidence *= self.confidence_decay_rate
            
        return self.target_belief
        
    def update_belief_with_observation(self, observation, observation_confidence=1.0):
        """
        Update belief using Bayesian inference with new observation
        
        Args:
            observation: Observed target position or None if not observed
            observation_confidence: Confidence in the observation (0-1)
            
        Returns:
            numpy.ndarray: Updated belief grid
        """
        if observation is None:
            # No observation, just decay confidence
            self.belief_confidence *= self.confidence_decay_rate
            return self.target_belief
            
        # Apply Bayes' rule: posterior ∝ likelihood × prior
        
        # 1. Create likelihood function (probability of observation given state)
        likelihood = np.zeros_like(self.target_belief)
        
        # Convert observation to grid coordinates
        obs_i, obs_j = self.world_to_grid(observation)
        
        # Create Gaussian likelihood centered at observation
        # with variance based on observation noise
        sigma = self.observation_noise_sigma / observation_confidence
        
        # Calculate likelihood for each grid cell
        for i in range(self.belief_grid_resolution[0]):
            for j in range(self.belief_grid_resolution[1]):
                dist_squared = (i - obs_i)**2 + (j - obs_j)**2
                likelihood[i, j] = np.exp(-dist_squared / (2 * sigma**2))
                
        # Normalize likelihood
        likelihood_sum = np.sum(likelihood)
        if likelihood_sum > 0:
            likelihood = likelihood / likelihood_sum
            
        # 2. Apply Bayes' rule
        posterior = self.target_belief * likelihood
        
        # Normalize posterior
        posterior_sum = np.sum(posterior)
        if posterior_sum > 0:
            posterior = posterior / posterior_sum
            
        # 3. Update belief
        # Weight new information by observation confidence
        self.target_belief = (1 - observation_confidence) * self.target_belief + \
                            observation_confidence * posterior
                            
        # Normalize again
        belief_sum = np.sum(self.target_belief)
        if belief_sum > 0:
            self.target_belief = self.target_belief / belief_sum
            
        # Update confidence based on new observation
        self.belief_confidence = max(self.belief_confidence, observation_confidence)
        
        return self.target_belief
        
    def fuse_beliefs(self, other_beliefs, other_confidences):
        """
        Fuse beliefs from multiple agents using weighted average
        
        Args:
            other_beliefs: List of belief grids from other agents
            other_confidences: List of confidence values for other beliefs
            
        Returns:
            numpy.ndarray: Fused belief grid
        """
        if not other_beliefs:
            return self.target_belief
            
        # Combine all beliefs including own
        all_beliefs = [self.target_belief] + other_beliefs
        all_confidences = [self.belief_confidence] + other_confidences
        
        # Convert confidences to weights
        weights = np.array(all_confidences)
        weights = weights / (np.sum(weights) + 1e-10)
        
        # Weighted average of beliefs
        fused_belief = np.zeros_like(self.target_belief)
        
        for i, belief in enumerate(all_beliefs):
            fused_belief += belief * weights[i]
            
        # Normalize
        belief_sum = np.sum(fused_belief)
        if belief_sum > 0:
            fused_belief = fused_belief / belief_sum
            
        # Update own belief if change is significant
        diff = np.sum(np.abs(fused_belief - self.target_belief))
        if diff > self.min_update_threshold:
            self.target_belief = fused_belief
            
            # Update confidence - use maximum of confidences to maintain consistency
            self.belief_confidence = max(all_confidences)
            
        return self.target_belief
        
    def get_highest_belief_position(self):
        """
        Get world position with highest belief probability
        
        Returns:
            tuple: (position, confidence)
        """
        # Find indices of maximum belief
        max_indices = np.unravel_index(np.argmax(self.target_belief), self.target_belief.shape)
        
        # Convert to world coordinates
        max_position = self.grid_to_world(max_indices)
        
        # Maximum belief value indicates certainty
        max_belief = self.target_belief[max_indices]
        
        # Adjust confidence based on peak belief value and overall confidence
        # Higher max value indicates more certainty in position
        position_confidence = self.belief_confidence * max_belief * np.prod(self.belief_grid_resolution)
        
        return max_position, position_confidence
        
    def update_obstacle_belief(self, obstacle_id, position, radius, confidence=1.0):
        """
        Update belief about obstacle position and size
        
        Args:
            obstacle_id: Unique identifier for obstacle
            position: Observed obstacle position
            radius: Observed obstacle radius
            confidence: Confidence in observation
            
        Returns:
            tuple: (estimated position, estimated radius, confidence)
        """
        if obstacle_id in self.obstacle_beliefs:
            # Existing obstacle, update beliefs
            prev_position, prev_radius, prev_confidence, _ = self.obstacle_beliefs[obstacle_id]
            
            # Weighted average based on confidence
            total_confidence = prev_confidence + confidence
            
            if total_confidence > 0:
                # Update position
                updated_position = (
                    prev_position * prev_confidence + position * confidence
                ) / total_confidence
                
                # Update radius
                updated_radius = (
                    prev_radius * prev_confidence + radius * confidence
                ) / total_confidence
                
                # Update confidence but enforce maximum
                updated_confidence = min(1.0, total_confidence)
                
                # Store timestamp
                timestamp = self.last_update_time
                
                self.obstacle_beliefs[obstacle_id] = (
                    updated_position, updated_radius, updated_confidence, timestamp
                )
                
                return updated_position, updated_radius, updated_confidence
            else:
                return prev_position, prev_radius, prev_confidence
        else:
            # New obstacle
            self.obstacle_beliefs[obstacle_id] = (
                position, radius, confidence, self.last_update_time
            )
            
            return position, radius, confidence
            
    def get_all_obstacle_beliefs(self):
        """
        Get all current obstacle beliefs
        
        Returns:
            list: List of (obstacle_id, position, radius, confidence) tuples
        """
        result = []
        
        for obstacle_id, (position, radius, confidence, timestamp) in self.obstacle_beliefs.items():
            # Decay confidence based on time since last update
            time_diff = self.last_update_time - timestamp
            decayed_confidence = confidence * (self.confidence_decay_rate ** time_diff)
            
            if decayed_confidence > 0.1:  # Only include if still somewhat confident
                result.append((obstacle_id, position, radius, decayed_confidence))
                
        return result
        
    def calculate_entropy(self):
        """
        Calculate entropy of current belief state
        
        Returns:
            float: Entropy value
        """
        # Filter out zeros to avoid log(0)
        valid_beliefs = self.target_belief[self.target_belief > 0]
        
        if len(valid_beliefs) == 0:
            return 0.0
            
        # Calculate Shannon entropy: -sum(p_i * log(p_i))
        entropy = -np.sum(valid_beliefs * np.log2(valid_beliefs))
        
        return entropy
        
    def calculate_belief_force(self, uav, nearby_uavs, current_time):
        """
        Calculate force based on current belief state
        
        Args:
            uav: The UAV to calculate force for
            nearby_uavs: List of nearby UAVs
            current_time: Current simulation time
            
        Returns:
            numpy.ndarray: The belief force vector [x, y]
        """
        # Update internal time
        self.last_update_time = current_time
        
        # Get estimated target position from belief
        target_position, position_confidence = self.get_highest_belief_position()
        
        # Direction to estimated target
        direction = target_position - uav.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
            
        # Calculate entropy of belief
        entropy = self.calculate_entropy()
        
        # Force magnitude based on confidence and entropy
        # Lower entropy (more certainty) and higher confidence means stronger force
        uncertainty_factor = 1.0 - (entropy / np.log2(np.prod(self.belief_grid_resolution)))
        
        # Scale down force when uncertainty is high
        force_magnitude = CONFIG["uav_max_acceleration"] * position_confidence * uncertainty_factor
        
        # Calculate force toward estimated target position
        belief_force = direction * force_magnitude
        
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(belief_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            belief_force = belief_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return belief_force
