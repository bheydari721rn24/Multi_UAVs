"""
Swarm Intelligence Module for Multi-UAV Systems

This module implements the Adaptive Hierarchical Federated Swarm Intelligence (AHFSI) framework
with advanced components including quantum-inspired optimization, information theory,
multi-scale temporal abstraction, game theory, and topological analysis.

Author: Research Team
Date: 2025
"""

import numpy as np
import math
from utils import CONFIG
import time

# Formation types and their ideal distances
FORMATION_DISTANCES = {
    "search": 2.0,      # Spread out to cover more area
    "approach": 1.5,    # Closer formation while approaching target
    "track": 1.2,       # Tight formation for tracking
    "encircle": 1.0,     # Equidistant for encirclement
    "capturing": 0.8,    # Very tight formation for capture
    "finished": 1.0      # Standard distance after capture
}

# ========================================================================================
# Core Swarm Behavior Classes
# ========================================================================================

class SwarmBehavior:
    """Base class for swarm behaviors implementing core functionality"""
    
    def __init__(self):
        """Initialize swarm behavior with configuration parameters"""
        self.config = CONFIG["swarm"]
        
    def calculate_cohesion_force(self, uav, nearby_uavs):
        """
        Calculate cohesion force to move toward center of nearby UAVs
        Uses adaptive weighting based on context
        
        Args:
            uav: The UAV for which to calculate the force
            nearby_uavs: List of nearby UAVs within communication range
            
        Returns:
            numpy.ndarray: The cohesion force vector [x, y]
        """
        if not nearby_uavs:
            return np.zeros(2)
            
        # Get behavior parameters
        behavior_config = self.config["behaviors"]["cohesion"]
        base_weight = behavior_config["weight"]
        distance_threshold = behavior_config["distance_threshold"]
        adaptive_factor = behavior_config["adaptive_factor"]
        
        # Get positions of nearby UAVs within threshold distance
        positions = []
        for other_uav in nearby_uavs:
            distance = np.linalg.norm(uav.position - other_uav.position)
            if distance <= distance_threshold and distance > 0:
                positions.append(other_uav.position)
                
        if not positions:
            return np.zeros(2)
            
        # Calculate center of mass
        center = np.mean(positions, axis=0)
        
        # Calculate distance to center
        distance_to_center = np.linalg.norm(center - uav.position)
        
        # Adjust weight adaptively based on distance
        # More weight when far from center, less when already close
        adaptive_weight = base_weight * (1.0 + adaptive_factor * 
                                       (distance_to_center / distance_threshold))
        
        # Calculate cohesion force toward center
        direction = (center - uav.position) / (distance_to_center + 1e-10)
        force = direction * adaptive_weight * CONFIG["uav_max_acceleration"]
        
        return force
        
    def calculate_separation_force(self, uav, nearby_uavs):
        """
        Calculate separation force to avoid collisions with other UAVs
        Applies stronger repulsion at closer distances
        
        Args:
            uav: The UAV for which to calculate the force
            nearby_uavs: List of nearby UAVs
            
        Returns:
            numpy.ndarray: The separation force vector [x, y]
        """
        if not nearby_uavs:
            return np.zeros(2)
            
        # Get behavior parameters
        behavior_config = self.config["behaviors"]["separation"]
        base_weight = behavior_config["weight"]
        min_distance = behavior_config["minimum_distance"]
        adaptive_factor = behavior_config["adaptive_factor"]
        
        separation_force = np.zeros(2)
        
        # Calculate repulsion from each nearby UAV
        for other_uav in nearby_uavs:
            if other_uav is uav:
                continue
                
            # Calculate distance and direction
            direction = uav.position - other_uav.position
            distance = np.linalg.norm(direction)
            
            # Apply separation only when too close
            if distance < min_distance and distance > 0:
                # Normalize direction
                direction = direction / distance
                
                # Stronger repulsion at closer distances (inverse square law)
                # Adjusted by adaptive factor for context-awareness
                repulsion_strength = base_weight * (1.0 + adaptive_factor * 
                                                 (1.0 - distance/min_distance)**2)
                
                # Use inverse square law for stronger effect at close distances
                repulsion = direction * repulsion_strength * (min_distance / distance)**2
                
                separation_force += repulsion
        
        # Scale force to max acceleration
        force_magnitude = np.linalg.norm(separation_force)
        if force_magnitude > 0:
            separation_force = separation_force / force_magnitude * CONFIG["uav_max_acceleration"]
        
        return separation_force
        
    def calculate_alignment_force(self, uav, nearby_uavs):
        """
        Calculate alignment force to match velocity with nearby UAVs
        
        Args:
            uav: The UAV for which to calculate the force
            nearby_uavs: List of nearby UAVs
            
        Returns:
            numpy.ndarray: The alignment force vector [x, y]
        """
        if not nearby_uavs:
            return np.zeros(2)
            
        # Get behavior parameters
        behavior_config = self.config["behaviors"]["alignment"]
        base_weight = behavior_config["weight"]
        max_distance = behavior_config["max_influence_distance"]
        adaptive_factor = behavior_config["adaptive_factor"]
        
        # Get velocities of nearby UAVs
        velocities = []
        weights = []
        
        for other_uav in nearby_uavs:
            if other_uav is uav:
                continue
                
            distance = np.linalg.norm(uav.position - other_uav.position)
            if distance <= max_distance:
                # Weight velocity by inverse distance
                weight = 1.0 / (distance + 0.1)
                velocities.append(other_uav.velocity)
                weights.append(weight)
                
        if not velocities:
            return np.zeros(2)
            
        # Calculate weighted average velocity
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        avg_velocity = np.zeros(2)
        for i, vel in enumerate(velocities):
            avg_velocity += vel * weights[i]
            
        # Calculate alignment force - tendency to match average velocity
        # Adjust based on contextual alignment factor
        velocity_diff = avg_velocity - uav.velocity
        
        # Adapt alignment weight based on current velocity alignment
        current_alignment = np.dot(uav.velocity, avg_velocity) / (
            np.linalg.norm(uav.velocity) * np.linalg.norm(avg_velocity) + 1e-10)
        
        # More alignment force when current alignment is low
        adaptive_weight = base_weight * (1.0 + adaptive_factor * (1.0 - current_alignment))
        
        alignment_force = velocity_diff * adaptive_weight
        
        # Scale force to max acceleration
        force_magnitude = np.linalg.norm(alignment_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            alignment_force = alignment_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return alignment_force
    
    def evaluate_formation_quality(self, uav_positions, current_position, task_state="search"):
        """
        Evaluate how well UAVs are maintaining the desired formation
        
        Args:
            uav_positions: List of positions of all UAVs
            current_position: Position of the UAV being evaluated
            task_state: Current task state (search, approach, track, etc.)
            
        Returns:
            float: Formation quality metric (0.0 to 1.0)
        """
        if len(uav_positions) < 2:  # Need at least 2 UAVs for any formation
            return 0.0
            
        # Get the ideal distance for current formation type
        if task_state in FORMATION_DISTANCES:
            ideal_distance = FORMATION_DISTANCES[task_state]
        else:
            ideal_distance = FORMATION_DISTANCES["search"]  # Default
            
        # Calculate distances to other UAVs
        distances = []
        for pos in uav_positions:
            if np.array_equal(pos, current_position):
                continue  # Skip self
            distance = np.linalg.norm(np.array(pos) - np.array(current_position))
            distances.append(distance)
            
        if not distances:
            return 0.0
            
        # Calculate how close distances are to ideal distance
        deviations = [abs(d - ideal_distance) / ideal_distance for d in distances]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Calculate spatial distribution metrics
        # Standard deviation of angles between UAVs indicates even spacing
        angles = []
        if len(uav_positions) >= 3:
            # Calculate center of mass excluding self
            other_positions = [pos for pos in uav_positions if not np.array_equal(pos, current_position)]
            center = np.mean(other_positions, axis=0)
            
            # Calculate angles from center to each UAV
            for pos in other_positions:
                vec = pos - center
                angle = math.atan2(vec[1], vec[0])
                angles.append(angle)
                
            # For even spatial distribution, angles should have low standard deviation
            if angles:
                # Normalize angles to handle the circular nature (wrapping around 2π)
                angle_diffs = []
                for i in range(len(angles)):
                    for j in range(i+1, len(angles)):
                        diff = abs(angles[i] - angles[j]) % (2 * math.pi)
                        if diff > math.pi:
                            diff = 2 * math.pi - diff
                        angle_diffs.append(diff)
                        
                if angle_diffs:
                    # Ideal distribution would have equal angles
                    ideal_angle_diff = 2 * math.pi / len(angles)
                    angle_deviations = [abs(diff - ideal_angle_diff) / ideal_angle_diff for diff in angle_diffs]
                    angle_quality = 1.0 - min(1.0, sum(angle_deviations) / len(angle_deviations))
                else:
                    angle_quality = 0.0
            else:
                angle_quality = 0.0
        else:
            angle_quality = 0.0
            
        # Overall formation quality combines distance and angle qualities
        distance_quality = 1.0 - min(1.0, avg_deviation)
        
        # Weight distance quality more for small groups, angle quality more for larger groups
        if len(uav_positions) <= 3:
            quality = 0.8 * distance_quality + 0.2 * angle_quality
        else:
            quality = 0.5 * distance_quality + 0.5 * angle_quality
            
        return quality
        
    def calculate_formation_force(self, uav, nearby_uavs, target_position=None, formation_type=None):
        """
        Calculate force to maintain formation with other UAVs
        
        Args:
            uav: The UAV for which to calculate the force
            nearby_uavs: List of nearby UAVs
            target_position: Position of target if available
            formation_type: Type of formation to maintain
            
        Returns:
            numpy.ndarray: The formation force vector [x, y]
        """
        if not nearby_uavs or not target_position:
            return np.zeros(2)
            
        # Get behavior parameters
        behavior_config = self.config["behaviors"]["formation"]
        base_weight = behavior_config["weight"]
        scale_factor = behavior_config["formation_scale_factor"]
        
        # If formation type not specified, choose based on context
        if formation_type is None:
            # Use different formations for different contexts
            # Circle for encirclement, line for blocking, etc.
            if hasattr(uav, 'role'):
                if uav.role == "ENCIRCLER":
                    formation_type = "circle"
                elif uav.role == "BLOCKER":
                    formation_type = "line"
                elif len(nearby_uavs) >= 2:
                    formation_type = "triangle"
                else:
                    formation_type = "line"
            else:
                formation_type = "circle"  # Default
                
        # All UAVs including self
        all_uavs = nearby_uavs + [uav]
        n_uavs = len(all_uavs)
        
        # Sort by ID for consistent positioning
        all_uavs.sort(key=lambda u: u.id)
        my_index = all_uavs.index(uav)
        
        formation_force = np.zeros(2)
        
        if formation_type == "circle":
            # Calculate position on circle around target
            angle = 2 * math.pi * my_index / n_uavs
            radius = scale_factor * CONFIG["capture_distance"] / 2
            
            target_x = target_position[0] + radius * math.cos(angle)
            target_y = target_position[1] + radius * math.sin(angle)
            target_pos = np.array([target_x, target_y])
            
            # Force toward target position on circle
            direction = target_pos - uav.position
            distance = np.linalg.norm(direction)
            
            # Stronger force when far from designated position
            force_magnitude = min(distance, CONFIG["uav_max_acceleration"])
            formation_force = direction / (distance + 1e-10) * force_magnitude * base_weight
            
        elif formation_type == "line":
            # Form line perpendicular to target direction
            direction = uav.position - target_position
            perp_direction = np.array([-direction[1], direction[0]])
            perp_direction = perp_direction / (np.linalg.norm(perp_direction) + 1e-10)
            
            spacing = scale_factor * 0.5
            offset = (my_index - n_uavs/2) * spacing
            
            target_pos = target_position + perp_direction * offset
            
            # Force toward target position in line
            direction = target_pos - uav.position
            distance = np.linalg.norm(direction)
            
            formation_force = direction / (distance + 1e-10) * min(distance, CONFIG["uav_max_acceleration"]) * base_weight
            
        elif formation_type == "triangle":
            # Form triangle with target at center
            if n_uavs < 3:
                return np.zeros(2)
                
            angles = [2 * math.pi * i / 3 + math.pi/6 for i in range(3)]
            radius = scale_factor * CONFIG["capture_distance"] / 2
            
            angle_index = my_index % 3
            target_x = target_position[0] + radius * math.cos(angles[angle_index])
            target_y = target_position[1] + radius * math.sin(angles[angle_index])
            target_pos = np.array([target_x, target_y])
            
            direction = target_pos - uav.position
            distance = np.linalg.norm(direction)
            
            formation_force = direction / (distance + 1e-10) * min(distance, CONFIG["uav_max_acceleration"]) * base_weight
            
        elif formation_type == "wedge":
            # V formation (good for approaching)
            leader_idx = 0
            
            if my_index == leader_idx:
                # Leader position
                target_pos = target_position  # Leader heads directly to target
            else:
                # Follower positions in V formation
                leader = all_uavs[leader_idx]
                leader_to_target = target_position - leader.position
                leader_direction = leader_to_target / (np.linalg.norm(leader_to_target) + 1e-10)
                
                # Calculate right perpendicular vector for V shape
                perp_direction = np.array([-leader_direction[1], leader_direction[0]])
                
                # Alternate sides of the V
                side = 1 if my_index % 2 == 1 else -1
                row = (my_index - 1) // 2 + 1
                
                # Position behind leader in V formation
                target_pos = leader.position - leader_direction * (row * 0.5 * scale_factor) + perp_direction * (side * row * 0.3 * scale_factor)
            
            direction = target_pos - uav.position
            distance = np.linalg.norm(direction)
            
            formation_force = direction / (distance + 1e-10) * min(distance, CONFIG["uav_max_acceleration"]) * base_weight
            
        return formation_force
