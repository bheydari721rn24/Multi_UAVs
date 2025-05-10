"""
Game-Theoretic Coordination Module for Multi-UAV Systems

This module implements Stackelberg game theory for leader-follower dynamics
and Nash equilibrium approaches for multi-agent coordination.

Author: Research Team
Date: 2025
"""

import numpy as np
import math
from utils import CONFIG

class GameTheoryCoordinator:
    """
    Game-theoretic coordination for multi-agent systems
    Implements Stackelberg leader-follower and Nash equilibrium approaches
    """
    
    def __init__(self):
        """Initialize game theory coordinator with configuration parameters"""
        self.config = CONFIG["swarm"]["game_theory"]
        self.leader_selection_interval = self.config["leader_selection_interval"]
        self.leadership_quality_threshold = self.config["leadership_quality_threshold"]
        self.follower_coherence_factor = self.config["follower_coherence_factor"]
        self.role_switching_cost = self.config["role_switching_cost"]
        self.stackelberg_alpha = self.config["stackelberg_alpha"]
        
        # State variables
        self.current_leader_id = None
        self.leadership_quality = {}
        self.role_counter = 0
        self.role_history = {}
        
    def select_leader(self, uavs, info_qualities):
        """
        Select a leader UAV based on information quality and position
        
        Args:
            uavs: List of UAV agents
            info_qualities: Dictionary mapping UAV IDs to information quality scores
            
        Returns:
            tuple: (leader UAV, followers list)
        """
        if not uavs:
            return None, []
            
        self.role_counter += 1
        
        # Only perform leader selection at specified intervals
        if self.role_counter % self.leader_selection_interval != 0:
            # Return current leader if available
            if self.current_leader_id is not None:
                leader = next((uav for uav in uavs if uav.id == self.current_leader_id), None)
                if leader:
                    followers = [uav for uav in uavs if uav.id != self.current_leader_id]
                    return leader, followers
            
        # Initialize leadership quality for new UAVs
        for uav in uavs:
            if uav.id not in self.leadership_quality:
                self.leadership_quality[uav.id] = 0.5  # Initial neutral quality
                
        # Calculate leadership quality based on information and position
        for uav in uavs:
            # Information quality component
            info_quality = info_qualities.get(uav.id, 0.5)
            
            # Positional quality - centrality to other UAVs
            total_distance = 0
            for other_uav in uavs:
                if other_uav.id != uav.id:
                    dist = np.linalg.norm(uav.position - other_uav.position)
                    total_distance += dist
                    
            # Lower average distance means more central position
            if len(uavs) > 1:
                avg_distance = total_distance / (len(uavs) - 1)
                positional_quality = np.exp(-avg_distance / 3.0)  # Normalize to [0,1]
            else:
                positional_quality = 1.0
                
            # Role stability component - penalize frequent switching
            stability = 1.0
            if uav.id in self.role_history:
                # How long since this UAV was last a leader
                time_since_leader = self.role_counter - self.role_history[uav.id]
                if time_since_leader < self.leader_selection_interval * 2:
                    # Recently was leader, penalize to avoid frequent switches
                    stability = 0.5
                    
            # Overall leadership quality
            new_quality = 0.4 * info_quality + 0.4 * positional_quality + 0.2 * stability
            
            # Smooth update of leadership quality
            if uav.id in self.leadership_quality:
                # Exponential moving average
                self.leadership_quality[uav.id] = 0.7 * self.leadership_quality[uav.id] + 0.3 * new_quality
            else:
                self.leadership_quality[uav.id] = new_quality
                
        # Find UAV with highest leadership quality
        best_quality = -1
        best_uav = None
        
        for uav in uavs:
            if self.leadership_quality[uav.id] > best_quality:
                best_quality = self.leadership_quality[uav.id]
                best_uav = uav
                
        # Check if best UAV meets leadership threshold
        if best_quality >= self.leadership_quality_threshold:
            leader = best_uav
            self.current_leader_id = leader.id
            self.role_history[leader.id] = self.role_counter
        else:
            # If no UAV meets threshold, rotate leadership 
            # Choose UAV that hasn't been leader for longest time
            leader_times = []
            for uav in uavs:
                last_time = self.role_history.get(uav.id, 0)
                leader_times.append((uav, self.role_counter - last_time))
                
            # Sort by time since last leadership (descending)
            leader_times.sort(key=lambda x: x[1], reverse=True)
            leader = leader_times[0][0]
            self.current_leader_id = leader.id
            self.role_history[leader.id] = self.role_counter
            
        # All other UAVs are followers
        followers = [uav for uav in uavs if uav.id != leader.id]
        
        return leader, followers
        
    def compute_leader_action(self, leader, followers, state):
        """
        Compute optimal action for leader in Stackelberg game
        
        Args:
            leader: Leader UAV
            followers: List of follower UAVs
            state: Current environment state
            
        Returns:
            numpy.ndarray: Optimal force vector for leader [x, y]
        """
        if not followers:
            # If no followers, act independently
            return np.zeros(2)
            
        # Extract relevant state information
        target_position = state.get("target_position", None)
        
        if target_position is None:
            # No target information, use standard behavior
            return np.zeros(2)
            
        # In Stackelberg game, leader anticipates followers' responses
        # Here we use a simplified model:
        # 1. Predict how followers will respond to leader movement
        # 2. Choose leader action that optimizes global objective
        
        # Leader's default action: move toward target
        direction_to_target = target_position - leader.position
        distance = np.linalg.norm(direction_to_target)
        
        if distance > 0:
            direction_to_target = direction_to_target / distance
            
        # Calculate average follower position
        follower_positions = np.array([f.position for f in followers])
        avg_follower_pos = np.mean(follower_positions, axis=0)
        
        # Check if followers are well-distributed around leader
        follower_directions = []
        for f in followers:
            rel_pos = f.position - leader.position
            rel_dist = np.linalg.norm(rel_pos)
            if rel_dist > 0:
                follower_directions.append(rel_pos / rel_dist)
                
        # Calculate coverage of follower directions (want them to be spread out)
        direction_coverage = 0.0
        if len(follower_directions) >= 2:
            for i in range(len(follower_directions)):
                for j in range(i+1, len(follower_directions)):
                    # Dot product close to -1 means nearly opposite directions (good coverage)
                    dot_product = np.dot(follower_directions[i], follower_directions[j])
                    direction_coverage += (1.0 - dot_product) / 2.0
                    
            # Normalize
            max_coverage = len(followers) * (len(followers) - 1) / 2
            direction_coverage /= max_coverage
        
        # Leader Stackelberg strategy:
        # 1. With good follower coverage: leader focuses on target
        # 2. With poor coverage: leader adjusts to improve formation
        
        # Calculate vector from leader to target
        target_vector = direction_to_target * CONFIG["uav_max_acceleration"]
        
        # Calculate vector to optimize formation (move to position that improves coverage)
        formation_vector = np.zeros(2)
        
        if direction_coverage < 0.7 and len(followers) >= 2:
            # Find direction with least coverage
            angle_coverage = np.zeros(8)  # 8 direction bins
            
            for dir_vec in follower_directions:
                angle = math.atan2(dir_vec[1], dir_vec[0])
                bin_idx = int((angle + math.pi) / (2 * math.pi / 8)) % 8
                angle_coverage[bin_idx] += 1
                
            # Find bin with minimum coverage
            min_bin = np.argmin(angle_coverage)
            
            # Calculate direction for that bin
            min_angle = min_bin * (2 * math.pi / 8) - math.pi
            min_direction = np.array([math.cos(min_angle), math.sin(min_angle)])
            
            # Leader should move slightly away from this least-covered direction
            formation_vector = -min_direction * CONFIG["uav_max_acceleration"] * 0.5
        
        # Combine objectives with weighting based on formation quality
        leader_force = target_vector * (0.5 + 0.5 * direction_coverage) + \
                      formation_vector * (0.5 - 0.5 * direction_coverage)
                      
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(leader_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            leader_force = leader_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return leader_force
        
    def compute_follower_action(self, follower, leader, state):
        """
        Compute optimal action for follower in Stackelberg game
        
        Args:
            follower: Follower UAV
            leader: Leader UAV
            state: Current environment state
            
        Returns:
            numpy.ndarray: Optimal force vector for follower [x, y]
        """
        if leader is None:
            # If no leader, act independently
            return np.zeros(2)
            
        # Extract relevant state information
        target_position = state.get("target_position", None)
        
        # In Stackelberg game, followers respond to leader's action
        # Balance between following leader and pursuing individual objectives
        
        # Vector from follower to leader
        leader_direction = leader.position - follower.position
        distance_to_leader = np.linalg.norm(leader_direction)
        
        if distance_to_leader > 0:
            leader_direction = leader_direction / distance_to_leader
            
        # Vector to target if available
        target_direction = np.zeros(2)
        if target_position is not None:
            target_vec = target_position - follower.position
            target_dist = np.linalg.norm(target_vec)
            if target_dist > 0:
                target_direction = target_vec / target_dist
                
        # Calculate desired formation position relative to leader
        # Distribute followers evenly around leader
        
        # Get follower's ID index among all followers (assume ascending order)
        follower_index = follower.id
        if hasattr(follower, 'follower_index'):
            follower_index = follower.follower_index
            
        # Calculate desired angle around leader
        num_followers = state.get("num_followers", 3)  # Default to 3 if not specified
        angle = 2 * math.pi * (follower_index % num_followers) / num_followers
        
        # Desired position is at fixed distance from leader at calculated angle
        formation_distance = CONFIG["capture_distance"] / 2
        desired_offset = np.array([
            formation_distance * math.cos(angle),
            formation_distance * math.sin(angle)
        ])
        
        # Leader's position plus offset gives desired formation position
        desired_position = leader.position + desired_offset
        
        # Direction to desired position
        formation_vec = desired_position - follower.position
        formation_dist = np.linalg.norm(formation_vec)
        
        if formation_dist > 0:
            formation_direction = formation_vec / formation_dist
        else:
            formation_direction = np.zeros(2)
            
        # Calculate follower force as weighted combination
        
        # Weight for formation increases with leader quality and when far from formation position
        formation_weight = self.follower_coherence_factor * (1.0 - np.exp(-formation_dist))
        
        # Weight for target decreases with distance to leader
        # When close to leader, more freedom to pursue target
        target_weight = (1.0 - self.follower_coherence_factor) * np.exp(-distance_to_leader / 3.0)
        
        # Follower Stackelberg strategy: respond optimally to leader's action
        # Weighted combination of formation maintenance and target pursuit
        follower_force = (
            formation_direction * formation_weight +
            target_direction * target_weight
        ) * CONFIG["uav_max_acceleration"]
        
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(follower_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            follower_force = follower_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return follower_force
        
    def compute_nash_equilibrium(self, uavs, state, utility_function):
        """
        Compute approximate Nash equilibrium for multi-agent coordination
        
        Args:
            uavs: List of UAV agents
            state: Current environment state
            utility_function: Function to calculate utility of joint action
            
        Returns:
            dict: Mapping of UAV IDs to optimal force vectors
        """
        if not uavs:
            return {}
            
        # Simplified Nash equilibrium calculation
        # In real implementation, would use numerical techniques like fictitious play
        
        # For demonstration, we'll implement a simplified approximate method
        # that maximizes joint utility while ensuring individual rationality
        
        # 1. Calculate individual "selfish" actions for each agent
        selfish_actions = {}
        for uav in uavs:
            # Simple selfish action: move toward target if visible
            target_position = state.get("target_position", None)
            if target_position is not None:
                direction = target_position - uav.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    selfish_action = direction / distance * CONFIG["uav_max_acceleration"]
                else:
                    selfish_action = np.zeros(2)
            else:
                selfish_action = np.zeros(2)
                
            selfish_actions[uav.id] = selfish_action
            
        # 2. Calculate joint action utility
        joint_utility = utility_function(uavs, selfish_actions, state)
        
        # 3. Iteratively adjust actions to improve joint utility
        max_iterations = 5
        learning_rate = 0.1
        
        for _ in range(max_iterations):
            for uav in uavs:
                # Try small perturbations of action in different directions
                best_action = selfish_actions[uav.id]
                best_utility = joint_utility
                
                # Check 8 directions for potential improvement
                for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    # Calculate perturbed action
                    perturbation = np.array([np.cos(angle), np.sin(angle)]) * learning_rate * CONFIG["uav_max_acceleration"]
                    perturbed_action = selfish_actions[uav.id] + perturbation
                    
                    # Ensure action doesn't exceed max acceleration
                    perturbed_magnitude = np.linalg.norm(perturbed_action)
                    if perturbed_magnitude > CONFIG["uav_max_acceleration"]:
                        perturbed_action = perturbed_action / perturbed_magnitude * CONFIG["uav_max_acceleration"]
                        
                    # Create temporary joint action to evaluate
                    temp_actions = selfish_actions.copy()
                    temp_actions[uav.id] = perturbed_action
                    
                    # Evaluate joint utility with this perturbation
                    temp_utility = utility_function(uavs, temp_actions, state)
                    
                    # Update best action if utility improved
                    if temp_utility > best_utility:
                        best_action = perturbed_action
                        best_utility = temp_utility
                        
                # Update action for this UAV
                selfish_actions[uav.id] = best_action
                
            # Update joint utility
            joint_utility = best_utility
            
        return selfish_actions
        
    def calculate_joint_utility(self, uavs, actions, state):
        """
        Calculate utility of joint action for Nash equilibrium
        
        Args:
            uavs: List of UAV agents
            actions: Dictionary mapping UAV IDs to actions
            state: Current environment state
            
        Returns:
            float: Joint utility value
        """
        if not uavs or not actions:
            return 0.0
            
        # Extract relevant state information
        target_position = state.get("target_position", None)
        
        if target_position is None:
            return 0.0
            
        # Components of joint utility
        
        # 1. Target pursuit utility (how well UAVs approach target)
        pursuit_utility = 0.0
        for uav in uavs:
            if uav.id in actions:
                # Direction from UAV to target
                target_dir = target_position - uav.position
                target_dist = np.linalg.norm(target_dir)
                
                if target_dist > 0:
                    target_dir = target_dir / target_dist
                    
                    # Dot product of action and target direction
                    # Higher when action aligns with direction to target
                    alignment = np.dot(actions[uav.id], target_dir)
                    
                    # Higher utility for actions toward target
                    pursuit_utility += max(0, alignment)
        
        # 2. Formation utility (how well UAVs maintain formation)
        formation_utility = 0.0
        if len(uavs) >= 2:
            # Calculate centroid of UAVs
            positions = np.array([uav.position for uav in uavs])
            centroid = np.mean(positions, axis=0)
            
            # Check if UAVs form a good encirclement around target
            encirclement_quality = 0.0
            
            # Calculate angles from target to UAVs
            angles = []
            for uav in uavs:
                relative_pos = uav.position - target_position
                angle = math.atan2(relative_pos[1], relative_pos[0])
                angles.append(angle)
                
            # Sort angles
            angles.sort()
            
            # Calculate gaps between angles
            angle_gaps = []
            for i in range(len(angles)):
                next_i = (i + 1) % len(angles)
                gap = (angles[next_i] - angles[i]) % (2 * math.pi)
                angle_gaps.append(gap)
                
            # Ideal gap
            ideal_gap = 2 * math.pi / len(uavs)
            
            # Calculate variance of gaps (lower is better)
            gap_variance = np.sum((np.array(angle_gaps) - ideal_gap)**2) / len(angle_gaps)
            
            # Convert variance to quality metric (higher is better)
            # Perfect encirclement has variance 0
            encirclement_quality = math.exp(-gap_variance * 2.0)
            
            # Final formation utility
            formation_utility = encirclement_quality
            
        # 3. Coordination utility (how well UAV actions coordinate)
        coordination_utility = 0.0
        if len(uavs) >= 2:
            # Check for conflicting actions
            for i, uav1 in enumerate(uavs):
                for j, uav2 in enumerate(uavs):
                    if i < j:
                        # Calculate future positions based on actions
                        future_pos1 = uav1.position + actions[uav1.id] * CONFIG["time_step"]
                        future_pos2 = uav2.position + actions[uav2.id] * CONFIG["time_step"]
                        
                        # Check proximity of future positions
                        future_dist = np.linalg.norm(future_pos1 - future_pos2)
                        
                        # Penalize if UAVs would get too close
                        min_safe_dist = 2 * CONFIG["uav_radius"]
                        if future_dist < min_safe_dist:
                            coordination_utility -= (min_safe_dist - future_dist) / min_safe_dist
                            
        # Combine components with weights
        joint_utility = (
            0.4 * pursuit_utility / len(uavs) +
            0.4 * formation_utility +
            0.2 * (1.0 + min(0, coordination_utility))  # Range [0, 1]
        )
        
        return joint_utility
        
    def calculate_game_theory_force(self, uav, nearby_uavs, state):
        """
        Calculate overall game-theoretic force for decision making
        
        Args:
            uav: The UAV to calculate force for
            nearby_uavs: List of nearby UAVs
            state: Current environment state
            
        Returns:
            numpy.ndarray: The game-theoretic force vector [x, y]
        """
        if not nearby_uavs:
            return np.zeros(2)
            
        # Calculate UAV information qualities
        info_qualities = {}
        for u in nearby_uavs + [uav]:
            if hasattr(u, 'belief_state') and isinstance(u.belief_state, dict):
                # Higher confidence means better information quality
                info_qualities[u.id] = u.belief_state.get("confidence", 0.5)
            else:
                info_qualities[u.id] = 0.5
                
        # Select leader and followers
        all_uavs = nearby_uavs + [uav]
        leader, followers = self.select_leader(all_uavs, info_qualities)
        
        # Calculate force based on role
        if leader and leader.id == uav.id:
            # This UAV is the leader
            stackelberg_force = self.compute_leader_action(uav, followers, state)
        elif leader:
            # This UAV is a follower
            stackelberg_force = self.compute_follower_action(uav, leader, state)
        else:
            stackelberg_force = np.zeros(2)
            
        # Also calculate Nash equilibrium for comparison
        state_with_info = dict(state)
        state_with_info["num_followers"] = len(followers)
        
        nash_actions = self.compute_nash_equilibrium(
            all_uavs, state_with_info, self.calculate_joint_utility)
            
        if uav.id in nash_actions:
            nash_force = nash_actions[uav.id]
        else:
            nash_force = np.zeros(2)
            
        # Combine Stackelberg and Nash forces with weighting
        # More weight to Stackelberg when leadership quality is high
        alpha = self.stackelberg_alpha
        if leader:
            alpha = max(alpha, self.leadership_quality.get(leader.id, alpha))
            
        game_theory_force = stackelberg_force * alpha + nash_force * (1 - alpha)
        
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(game_theory_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            game_theory_force = game_theory_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return game_theory_force
