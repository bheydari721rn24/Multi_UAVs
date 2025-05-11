"""
Information-Theoretic Decision Framework for Multi-UAV Systems

This module implements an entropy-based decision framework that optimizes 
information gain for swarm coordination and target tracking.

Author: Research Team
Date: 2025
"""

import numpy as np
import math
from utils import CONFIG
from scipy.stats import entropy as scipy_entropy


class InformationTheory:
    """
    Information-theoretic decision making for swarm systems
    Uses entropy and mutual information to guide decision making
    """
    
    def __init__(self, scenario_width=CONFIG["scenario_width"], 
                 scenario_height=CONFIG["scenario_height"]):
        """Initialize information theory framework with configuration parameters"""
        self.config = CONFIG["swarm"]["information_theory"]
        self.grid_size = self.config["entropy_grid_size"]
        self.scenario_width = scenario_width
        self.scenario_height = scenario_height
        
        # Initialize entropy grid - higher values = more uncertainty
        self.entropy_grid = np.ones(self.grid_size) * 0.5
        self.entropy_decay_rate = self.config["entropy_decay_rate"]
        self.information_gain_weight = self.config["information_gain_weight"]
        self.exploration_factor = self.config["exploration_factor"]
        self.mutual_info_threshold = self.config["mutual_information_threshold"]
        
        # Cell size in world coordinates
        self.cell_width = scenario_width / self.grid_size[0]
        self.cell_height = scenario_height / self.grid_size[1]
    
    def calculate_coverage_metric(self, uav_positions):
        """
        Calculate coverage efficiency metric for the UAV swarm
        
        Args:
            uav_positions: List of UAV positions
            
        Returns:
            float: Coverage metric (0.0 to 1.0), higher is better
        """
        if not uav_positions or len(uav_positions) < 1:
            return 0.0
            
        # Convert positions to grid coordinates
        grid_positions = []
        for pos in uav_positions:
            # Convert from world coordinates to grid indices
            grid_x = int(pos[0] / self.cell_width)
            grid_y = int(pos[1] / self.cell_height)
            
            # Ensure within bounds
            grid_x = max(0, min(grid_x, self.grid_size[0] - 1))
            grid_y = max(0, min(grid_y, self.grid_size[1] - 1))
            
            grid_positions.append((grid_x, grid_y))
        
        # Sensor range in grid cells
        sensor_range_cells = int(CONFIG["sensor_range"] / max(self.cell_width, self.cell_height))
        
        # Calculate observed cells
        observed_cells = set()
        for x, y in grid_positions:
            # Add all cells within sensor range
            for dx in range(-sensor_range_cells, sensor_range_cells + 1):
                for dy in range(-sensor_range_cells, sensor_range_cells + 1):
                    # Calculate distance in grid cells
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= sensor_range_cells:
                        # Add cell to observed set if within grid boundaries
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                            observed_cells.add((nx, ny))
        
        # Calculate coverage percentage
        total_cells = self.grid_size[0] * self.grid_size[1]
        coverage_percentage = len(observed_cells) / total_cells
        
        # Calculate spatial distribution coefficient
        if len(uav_positions) >= 2:
            # Calculate average distance between UAVs
            total_distance = 0
            count = 0
            for i in range(len(uav_positions)):
                for j in range(i+1, len(uav_positions)):
                    dist = np.linalg.norm(np.array(uav_positions[i]) - np.array(uav_positions[j]))
                    total_distance += dist
                    count += 1
            
            avg_distance = total_distance / count if count > 0 else 0
            
            # Normalize by diagonal distance of environment
            max_distance = math.sqrt(self.scenario_width**2 + self.scenario_height**2)
            normalized_distance = min(1.0, avg_distance / (max_distance * 0.5))
            
            # Optimal distribution has an intermediate value - not too close, not too far
            # Use a Gaussian curve centered at 0.4
            distribution_quality = math.exp(-((normalized_distance - 0.4) ** 2) / 0.1)
        else:
            distribution_quality = 0.5  # Default for single UAV
        
        # Combine metrics with appropriate weighting
        coverage_metric = 0.7 * coverage_percentage + 0.3 * distribution_quality
        
        return coverage_metric
    
    def update_entropy_grid(self, uav_positions, target_position=None, observed_cells=None):
        """
        Update entropy grid based on UAV observations and time decay
        
        Args:
            uav_positions: List of UAV positions
            target_position: Position of target if observed
            observed_cells: List of grid cells directly observed by sensors
            
        Returns:
            numpy.ndarray: Updated entropy grid
        """
        # Apply time decay - uncertainty increases over time
        self.entropy_grid = self.entropy_grid * self.entropy_decay_rate
        
        # Bound entropy values
        self.entropy_grid = np.clip(self.entropy_grid, 0.0, 1.0)
        
        # Update entropy for cells containing UAVs - uncertainty decreases
        for pos in uav_positions:
            # Convert world position to grid indices
            grid_i = min(int(pos[0] / self.cell_width), self.grid_size[0] - 1)
            grid_j = min(int(pos[1] / self.cell_height), self.grid_size[1] - 1)
            
            # Decrease entropy in observed cell and neighbors
            for i in range(max(0, grid_i-1), min(self.grid_size[0], grid_i+2)):
                for j in range(max(0, grid_j-1), min(self.grid_size[1], grid_j+2)):
                    # Calculate distance-based decay
                    dist = np.sqrt((i - grid_i)**2 + (j - grid_j)**2)
                    decay = np.exp(-dist * 2)
                    
                    # Reduce entropy based on observation
                    self.entropy_grid[i, j] = max(0.0, self.entropy_grid[i, j] - 0.1 * decay)
        
        # If target observed, update entropy at target position
        if target_position is not None:
            grid_i = min(int(target_position[0] / self.cell_width), self.grid_size[0] - 1)
            grid_j = min(int(target_position[1] / self.cell_height), self.grid_size[1] - 1)
            
            # Set low entropy at target position
            for i in range(max(0, grid_i-1), min(self.grid_size[0], grid_i+2)):
                for j in range(max(0, grid_j-1), min(self.grid_size[1], grid_j+2)):
                    self.entropy_grid[i, j] = 0.1
                    
        # Update entropy for observed cells from sensor data
        if observed_cells is not None:
            for cell in observed_cells:
                i, j = cell
                if 0 <= i < self.grid_size[0] and 0 <= j < self.grid_size[1]:
                    self.entropy_grid[i, j] = max(0.0, self.entropy_grid[i, j] - 0.2)
        
        return self.entropy_grid
        
    def calculate_information_gain(self, position, other_positions):
        """
        Calculate expected information gain from moving to a position
        
        Args:
            position: Position to evaluate [x, y]
            other_positions: Positions of other UAVs
            
        Returns:
            float: Expected information gain
        """
        # Convert position to grid indices
        grid_i = min(int(position[0] / self.cell_width), self.grid_size[0] - 1)
        grid_j = min(int(position[1] / self.cell_height), self.grid_size[1] - 1)
        
        # Current entropy at position
        current_entropy = self.entropy_grid[grid_i, grid_j]
        
        # Information gain is higher in high-entropy areas
        individual_gain = current_entropy
        
        # Calculate joint entropy with other UAVs
        joint_positions = list(other_positions) + [position]
        
        # Convert all positions to grid coordinates
        grid_positions = []
        for pos in joint_positions:
            i = min(int(pos[0] / self.cell_width), self.grid_size[0] - 1)
            j = min(int(pos[1] / self.cell_height), self.grid_size[1] - 1)
            grid_positions.append((i, j))
            
        # Calculate joint information gain
        observed_entropies = []
        for i, j in grid_positions:
            observed_entropies.append(self.entropy_grid[i, j])
        
        # Estimate joint information gain
        joint_gain = np.sum(observed_entropies)
        
        # Correct for overlapping information (avoid duplicate counting)
        overlap = self.calculate_mutual_information(grid_positions)
        
        # Final information gain with correction
        info_gain = individual_gain + (joint_gain - overlap) * self.information_gain_weight
        
        return max(0, info_gain)
        
    def calculate_mutual_information(self, grid_positions):
        """
        Calculate mutual information between positions
        
        Args:
            grid_positions: List of grid positions [(i,j), ...]
            
        Returns:
            float: Estimated mutual information
        """
        if len(grid_positions) <= 1:
            return 0.0
            
        # Calculate pairwise distances between positions
        mutual_info = 0.0
        
        for i in range(len(grid_positions)):
            for j in range(i+1, len(grid_positions)):
                pos1 = grid_positions[i]
                pos2 = grid_positions[j]
                
                # Calculate grid distance
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                # Closer positions have higher mutual information (more redundancy)
                if dist < 3:  # Within 3 grid cells
                    # Mutual information decreases with distance
                    mi = np.exp(-dist/2) * min(self.entropy_grid[pos1], self.entropy_grid[pos2])
                    mutual_info += mi
        
        return mutual_info
        
    def interpolate_entropy(self, position, entropy_grid=None):
        """
        Interpolate entropy at a specific world position
        
        Args:
            position: World position [x, y]
            entropy_grid: Entropy grid to use, defaults to self.entropy_grid
            
        Returns:
            float: Interpolated entropy value
        """
        if entropy_grid is None:
            entropy_grid = self.entropy_grid
            
        # Convert position to grid coordinates
        grid_x = position[0] / self.cell_width
        grid_y = position[1] / self.cell_height
        
        # Get integer grid indices
        i = int(grid_x)
        j = int(grid_y)
        
        # Bound indices to grid dimensions
        i = max(0, min(i, self.grid_size[0] - 2))
        j = max(0, min(j, self.grid_size[1] - 2))
        
        # Get fractional parts for interpolation
        dx = grid_x - i
        dy = grid_y - j
        
        # Bilinear interpolation
        value = (1-dx)*(1-dy)*entropy_grid[i, j] + \
                dx*(1-dy)*entropy_grid[i+1, j] + \
                (1-dx)*dy*entropy_grid[i, j+1] + \
                dx*dy*entropy_grid[i+1, j+1]
                
        return value
        
    def calculate_exploration_force(self, uav, nearby_uavs):
        """
        Calculate force for exploration based on entropy
        
        Args:
            uav: The UAV to calculate force for
            nearby_uavs: List of nearby UAVs
            
        Returns:
            numpy.ndarray: The exploration force vector [x, y]
        """
        # Get positions of nearby UAVs
        other_positions = [other_uav.position for other_uav in nearby_uavs 
                          if other_uav is not uav]
        
        # Calculate gradient of information gain in 4 directions
        position = uav.position
        step_size = 0.5  # Step size for gradient calculation
        
        pos_x_plus = np.array([position[0] + step_size, position[1]])
        pos_x_minus = np.array([position[0] - step_size, position[1]])
        pos_y_plus = np.array([position[0], position[1] + step_size])
        pos_y_minus = np.array([position[0], position[1] - step_size])
        
        # Calculate information gain in each direction
        gain_x_plus = self.calculate_information_gain(pos_x_plus, other_positions)
        gain_x_minus = self.calculate_information_gain(pos_x_minus, other_positions)
        gain_y_plus = self.calculate_information_gain(pos_y_plus, other_positions)
        gain_y_minus = self.calculate_information_gain(pos_y_minus, other_positions)
        
        # Calculate information gradient
        grad_x = (gain_x_plus - gain_x_minus) / (2 * step_size)
        grad_y = (gain_y_plus - gain_y_minus) / (2 * step_size)
        
        info_gradient = np.array([grad_x, grad_y])
        
        # Normalize gradient
        gradient_norm = np.linalg.norm(info_gradient)
        if gradient_norm > 1e-10:
            info_gradient = info_gradient / gradient_norm
            
        # Calculate exploration force
        exploration_force = info_gradient * self.exploration_factor * CONFIG["uav_max_acceleration"]
        
        # Add random exploration component
        random_direction = np.random.randn(2)
        random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-10)
        
        # Random component decreases as we have more information
        current_entropy = self.interpolate_entropy(position)
        random_weight = current_entropy * 0.3
        
        exploration_force += random_direction * random_weight * CONFIG["uav_max_acceleration"]
        
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(exploration_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            exploration_force = exploration_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return exploration_force
        
    def calculate_coverage_metric(self, uav_positions):
        """
        Calculate how well UAVs cover the high-entropy regions
        
        Args:
            uav_positions: List of UAV positions
            
        Returns:
            float: Coverage metric (0-1)
        """
        # Create coverage grid
        coverage = np.zeros_like(self.entropy_grid)
        
        # For each UAV, mark covered areas
        for pos in uav_positions:
            # Convert to grid indices
            grid_i = min(int(pos[0] / self.cell_width), self.grid_size[0] - 1)
            grid_j = min(int(pos[1] / self.cell_height), self.grid_size[1] - 1)
            
            # Mark cells as covered with distance-based decay
            sensor_range_grid = int(CONFIG["sensor_range"] / self.cell_width) + 1
            
            for i in range(max(0, grid_i-sensor_range_grid), 
                          min(self.grid_size[0], grid_i+sensor_range_grid+1)):
                for j in range(max(0, grid_j-sensor_range_grid), 
                              min(self.grid_size[1], grid_j+sensor_range_grid+1)):
                    # Calculate distance
                    dist = np.sqrt((i - grid_i)**2 + (j - grid_j)**2) * self.cell_width
                    
                    if dist <= CONFIG["sensor_range"]:
                        # Mark as covered with distance-based decay
                        coverage_value = 1.0 - dist/CONFIG["sensor_range"]
                        coverage[i, j] = max(coverage[i, j], coverage_value)
        
        # Calculate entropy-weighted coverage
        weighted_coverage = coverage * self.entropy_grid
        
        # Calculate coverage metric
        total_entropy = np.sum(self.entropy_grid)
        covered_entropy = np.sum(weighted_coverage)
        
        if total_entropy > 0:
            coverage_metric = covered_entropy / total_entropy
        else:
            coverage_metric = 1.0  # Perfect coverage if no uncertainty
            
        return coverage_metric
        
    def calculate_entropy_of_distribution(self, probabilities):
        """
        Calculate Shannon entropy of a probability distribution
        
        Args:
            probabilities: Array of probabilities (must sum to 1)
            
        Returns:
            float: Shannon entropy
        """
        # Filter out zeros to avoid log(0)
        valid_probs = probabilities[probabilities > 0]
        
        if len(valid_probs) == 0:
            return 0.0
            
        # Calculate Shannon entropy: -sum(p_i * log(p_i))
        return -np.sum(valid_probs * np.log2(valid_probs))
        
    def calculate_kl_divergence(self, p, q):
        """
        Calculate Kullback-Leibler divergence between distributions
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            float: KL divergence
        """
        # Ensure valid probability distributions
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        return scipy_entropy(p, q)
        
    def calculate_jensen_shannon_divergence(self, p, q):
        """
        Calculate Jensen-Shannon divergence (symmetric measure)
        
        Args:
            p: First probability distribution
            q: Second probability distribution
            
        Returns:
            float: JS divergence
        """
        # Ensure valid probability distributions
        p = np.maximum(p, 1e-10)
        q = np.maximum(q, 1e-10)
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate mid-point distribution
        m = 0.5 * (p + q)
        
        # Calculate JS divergence
        return 0.5 * (self.calculate_kl_divergence(p, m) + 
                     self.calculate_kl_divergence(q, m))
                     
    def calculate_information_force(self, uav, nearby_uavs, target_position=None):
        """
        Calculate overall information-theoretic force for decision making
        
        Args:
            uav: The UAV to calculate force for
            nearby_uavs: List of nearby UAVs
            target_position: Position of target if known
            
        Returns:
            numpy.ndarray: The information force vector [x, y]
        """
        # Calculate exploration force based on entropy
        exploration_force = self.calculate_exploration_force(uav, nearby_uavs)
        
        # Initialize target-based force
        target_force = np.zeros(2)
        
        # If target known, balance exploration vs exploitation
        if target_position is not None:
            # Direction to target
            direction = target_position - uav.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Target force (exploitation)
                target_force = direction / distance * CONFIG["uav_max_acceleration"]
                
                # Balance between exploration and exploitation based on distance
                # More exploration when far, more exploitation when close
                exploitation_weight = min(1.0, CONFIG["capture_distance"] / max(distance, 0.1))
                exploration_weight = 1.0 - exploitation_weight
                
                information_force = (
                    exploration_force * exploration_weight +
                    target_force * exploitation_weight
                )
            else:
                information_force = exploration_force
        else:
            # Pure exploration when target unknown
            information_force = exploration_force
            
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(information_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            information_force = information_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return information_force
