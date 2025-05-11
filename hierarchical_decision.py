"""
Hierarchical Decision Framework for Multi-UAV Systems

This module implements multi-scale temporal abstraction for decision making,
allowing UAVs to plan and coordinate at different time scales.

Author: Research Team
Date: 2025
"""

import numpy as np
from utils import CONFIG


class TemporalHierarchyController:
    """
    Multi-scale temporal abstraction for hierarchical decision making
    Enables planning at different time horizons simultaneously
    """
    
    def __init__(self, num_scales=None):
        """Initialize temporal hierarchy controller with configuration parameters"""
        self.config = CONFIG["swarm"]["temporal_abstraction"]
        
        if num_scales is None:
            self.num_scales = self.config["num_temporal_scales"]
        else:
            self.num_scales = num_scales
            
        self.temporal_horizons = self.config["temporal_horizons"]
        if len(self.temporal_horizons) < self.num_scales:
            # Extend horizons if not enough provided
            self.temporal_horizons = self.temporal_horizons + [
                self.temporal_horizons[-1] * 2] * (self.num_scales - len(self.temporal_horizons))
            
        self.scale_update_rates = self.config["scale_update_rates"]
        if len(self.scale_update_rates) < self.num_scales:
            # Extend update rates if not enough provided
            self.scale_update_rates = self.scale_update_rates + [
                self.scale_update_rates[-1] * 0.5] * (self.num_scales - len(self.scale_update_rates))
            
        self.discount_factors = self.config["temporal_discount_factors"]
        if len(self.discount_factors) < self.num_scales:
            # Extend discount factors if not enough provided
            self.discount_factors = self.discount_factors + [
                max(0.99, self.discount_factors[-1] + 0.01)] * (self.num_scales - len(self.discount_factors))
            
        # Initialize goals at different temporal scales
        self.temporal_goals = [None] * self.num_scales
        
        # Counters for updates at each scale
        self.update_counters = [0] * self.num_scales
        
        # Value estimates at each scale
        self.value_estimates = [0.0] * self.num_scales
        
        # Action preferences at each scale
        self.action_preferences = [np.zeros(2)] * self.num_scales
        
    def update(self, state, reward):
        """
        Update temporal hierarchy with new state and reward
        
        Args:
            state: Current environment state
            reward: Reward received
            
        Returns:
            list: Indices of temporal scales that were updated
        """
        updated_scales = []
        
        for i in range(self.num_scales):
            # Update value estimate with reward and discount factor
            self.value_estimates[i] = reward + self.discount_factors[i] * self.value_estimates[i]
            
            # Increment counter for this scale
            self.update_counters[i] += 1
            
            # Check if update needed at this scale
            if np.random.rand() < self.scale_update_rates[i]:
                self.update_counters[i] = 0
                self.temporal_goals[i] = self.compute_goal_at_scale(state, i)
                updated_scales.append(i)
                
        return updated_scales
        
    def compute_goal_at_scale(self, state, scale):
        """
        Compute goal at the given temporal scale
        
        Args:
            state: Current environment state
            scale: Temporal scale index (0=shortest, higher=longer)
            
        Returns:
            dict: Goal specification for this scale
        """
        if scale == 0:  # Shortest scale - tactical decisions
            return self.compute_tactical_goal(state)
        elif scale == 1:  # Medium scale - operational decisions
            return self.compute_operational_goal(state)
        else:  # Longest scale - strategic decisions
            return self.compute_strategic_goal(state)
            
    def compute_tactical_goal(self, state):
        """
        Compute short-term tactical goal (e.g., immediate movement)
        
        Args:
            state: Current environment state
            
        Returns:
            dict: Tactical goal specification
        """
        # Extract relevant state information
        target_visible = state.get("target_visible", False)
        target_position = state.get("target_position", None)
        obstacle_positions = state.get("obstacle_positions", [])
        
        if target_visible and target_position is not None:
            # When target visible, tactical goal is to move toward it
            return {
                "type": "approach",
                "position": target_position,
                "priority": 0.8,
                "time_horizon": self.temporal_horizons[0]
            }
        elif obstacle_positions:
            # When obstacles nearby, tactical goal is to avoid the closest one
            closest_obstacle = min(obstacle_positions, 
                                  key=lambda p: np.linalg.norm(p - state["position"]))
            return {
                "type": "avoid",
                "position": closest_obstacle,
                "priority": 0.9,
                "time_horizon": self.temporal_horizons[0]
            }
        else:
            # Default tactical goal is to maintain formation
            return {
                "type": "formation",
                "priority": 0.7,
                "time_horizon": self.temporal_horizons[0]
            }
            
    def compute_operational_goal(self, state):
        """
        Compute medium-term operational goal (e.g., encirclement, tracking)
        
        Args:
            state: Current environment state
            
        Returns:
            dict: Operational goal specification
        """
        # Extract relevant state information
        target_visible = state.get("target_visible", False)
        target_position = state.get("target_position", None)
        target_velocity = state.get("target_velocity", np.zeros(2))
        nearby_uavs = state.get("nearby_uavs", [])
        
        if target_visible and target_position is not None:
            if len(nearby_uavs) >= 2:
                # With multiple UAVs and visible target, operational goal is encirclement
                return {
                    "type": "encircle",
                    "position": target_position,
                    "velocity": target_velocity,
                    "priority": 0.85,
                    "time_horizon": self.temporal_horizons[1]
                }
            else:
                # With single UAV and visible target, operational goal is tracking
                return {
                    "type": "track",
                    "position": target_position,
                    "velocity": target_velocity,
                    "priority": 0.8,
                    "time_horizon": self.temporal_horizons[1]
                }
        else:
            # Default operational goal is to search
            return {
                "type": "search",
                "priority": 0.75,
                "time_horizon": self.temporal_horizons[1]
            }
            
    def compute_strategic_goal(self, state):
        """
        Compute long-term strategic goal (e.g., area coverage, capture)
        
        Args:
            state: Current environment state
            
        Returns:
            dict: Strategic goal specification
        """
        # Extract relevant state information
        target_visible = state.get("target_visible", False)
        target_position = state.get("target_position", None)
        target_history = state.get("target_history", [])
        scenario_center = np.array([CONFIG["scenario_width"], CONFIG["scenario_height"]]) / 2
        
        if target_visible and target_position is not None:
            # Strategic goal with visible target is capture
            return {
                "type": "capture",
                "position": target_position,
                "priority": 0.9,
                "time_horizon": self.temporal_horizons[2]
            }
        elif target_history:
            # Predict target movement from history
            recent_positions = target_history[-min(10, len(target_history)):]
            if len(recent_positions) >= 2:
                # Linear prediction of future position
                direction = recent_positions[-1] - recent_positions[0]
                predicted_position = recent_positions[-1] + direction
                
                # Bound prediction to scenario
                predicted_position[0] = max(0, min(CONFIG["scenario_width"], predicted_position[0]))
                predicted_position[1] = max(0, min(CONFIG["scenario_height"], predicted_position[1]))
                
                return {
                    "type": "intercept",
                    "position": predicted_position,
                    "priority": 0.8,
                    "time_horizon": self.temporal_horizons[2]
                }
        
        # Default strategic goal is to patrol/cover key areas
        return {
            "type": "patrol",
            "position": scenario_center,
            "priority": 0.7,
            "time_horizon": self.temporal_horizons[2]
        }
        
    def integrate_goals(self, state):
        """
        Integrate goals from all temporal scales
        
        Args:
            state: Current environment state
            
        Returns:
            dict: Integrated goal specification
        """
        # Filter out None goals
        valid_goals = [goal for goal in self.temporal_goals if goal is not None]
        
        if not valid_goals:
            # No valid goals yet
            return {
                "type": "default",
                "priority": 0.5
            }
            
        # Compute total priority
        total_priority = sum(goal["priority"] for goal in valid_goals)
        
        # Compute weighted goal attributes
        goal_type = max(valid_goals, key=lambda g: g["priority"])["type"]
        
        # For position-based goals, compute weighted position
        if any("position" in goal for goal in valid_goals):
            position_goals = [goal for goal in valid_goals if "position" in goal]
            
            if position_goals:
                weighted_position = np.zeros(2)
                position_weight = 0.0
                
                for goal in position_goals:
                    weighted_position += goal["position"] * goal["priority"]
                    position_weight += goal["priority"]
                    
                if position_weight > 0:
                    weighted_position /= position_weight
                    
                integrated_goal = {
                    "type": goal_type,
                    "position": weighted_position,
                    "priority": total_priority / len(valid_goals)
                }
            else:
                integrated_goal = {
                    "type": goal_type,
                    "priority": total_priority / len(valid_goals)
                }
        else:
            integrated_goal = {
                "type": goal_type,
                "priority": total_priority / len(valid_goals)
            }
            
        return integrated_goal
        
    def calculate_goal_force(self, uav, integrated_goal):
        """
        Calculate force to achieve the integrated goal
        
        Args:
            uav: The UAV to calculate force for
            integrated_goal: The integrated goal
            
        Returns:
            numpy.ndarray: The goal force vector [x, y]
        """
        goal_type = integrated_goal["type"]
        
        if goal_type in ["approach", "track", "capture", "intercept"] and "position" in integrated_goal:
            # Position-based goals - move toward position
            target_position = integrated_goal["position"]
            direction = target_position - uav.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Force toward goal position, scaled by priority
                force = direction / distance * CONFIG["uav_max_acceleration"] * integrated_goal["priority"]
                return force
                
        elif goal_type == "avoid" and "position" in integrated_goal:
            # Avoidance goal - move away from position
            obstacle_position = integrated_goal["position"]
            direction = uav.position - obstacle_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Force away from obstacle, stronger when closer
                repulsion_strength = CONFIG["uav_max_acceleration"] * integrated_goal["priority"] * (
                    2.0 / (distance + 0.5))
                force = direction / distance * repulsion_strength
                return force
                
        elif goal_type == "encircle" and "position" in integrated_goal:
            # Calculate position on circle around target
            target_position = integrated_goal["position"]
            direction = uav.position - target_position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Normalize to unit vector
                direction = direction / distance
                
                # Target distance for encirclement
                target_distance = CONFIG["capture_distance"] / 2
                
                # Tangential component (perpendicular to radial direction)
                tangent = np.array([-direction[1], direction[0]])
                
                # Radial component (to maintain distance)
                radial_force = (target_distance - distance) * direction
                
                # Combined force with more weight to tangential component for circling
                force = (radial_force * 0.3 + tangent * 0.7) * CONFIG["uav_max_acceleration"] * integrated_goal["priority"]
                return force
                
        elif goal_type == "search" or goal_type == "patrol":
            # Generate exploratory force - use gradient of value function or random direction
            random_direction = np.random.randn(2)
            random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-10)
            
            force = random_direction * CONFIG["uav_max_acceleration"] * integrated_goal["priority"] * 0.7
            return force
            
        elif goal_type == "formation":
            # Formation maintenance handled by swarm behaviors
            # Just return small centering force to scenario center
            center = np.array([CONFIG["scenario_width"], CONFIG["scenario_height"]]) / 2
            direction = center - uav.position
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                force = direction / distance * CONFIG["uav_max_acceleration"] * 0.1
                return force
        
        # Default: small random force
        random_direction = np.random.randn(2)
        if np.linalg.norm(random_direction) > 0:
            random_direction = random_direction / np.linalg.norm(random_direction)
        return random_direction * CONFIG["uav_max_acceleration"] * 0.1
        
    def calculate_hierarchical_force(self, uav, state):
        """
        Calculate overall hierarchical force for decision making
        
        Args:
            uav: The UAV to calculate force for
            state: Current environment state
            
        Returns:
            numpy.ndarray: The hierarchical force vector [x, y]
        """
        # Integrate goals from all temporal scales
        integrated_goal = self.integrate_goals(state)
        
        # Calculate force to achieve the integrated goal
        hierarchical_force = self.calculate_goal_force(uav, integrated_goal)
        
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(hierarchical_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            hierarchical_force = hierarchical_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return hierarchical_force
