import numpy as np
import math
import time
import os
from datetime import datetime
from agents import UAV, Target, Obstacle
from utils import CONFIG, calculate_triangle_area

# Try to import the logger, if not available, create a dummy logger
try:
    from logging_util import SimulationLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    print("Warning: logging_util module not found. Running without detailed logging.")
    
    # Create a dummy logger class
    class DummyLogger:
        def __init__(self, *args, **kwargs):
            pass
            
        def log_config(self, *args, **kwargs):
            pass
            
        def log_episode_start(self, *args, **kwargs):
            pass
            
        def log_episode_end(self, *args, **kwargs):
            pass
            
        def log_step(self, *args, **kwargs):
            pass
            
        def log_capture(self, *args, **kwargs):
            pass
            
        def log_error(self, *args, **kwargs):
            print(f"ERROR: {args[0]}")
            
        def get_summary_stats(self):
            return "Logging disabled"
            
        def save_summary(self):
            pass

class Environment:
    def __init__(self, num_uavs=3, num_obstacles=3, dynamic_obstacles=False, seed=None, enable_logging=True):
        self.num_uavs = num_uavs
        self.num_obstacles = num_obstacles
        self.dynamic_obstacles = dynamic_obstacles
        self.uavs = []
        self.target = None
        self.obstacles = []
        self.step_count = 0
        self._prev_distances = None  # For calculating reward changes
        self.task_state = None
        self.width = CONFIG["scenario_width"]
        self.height = CONFIG["scenario_height"]
        self.episode_num = 0  # Added episode_num variable
        
        # Seed management
        self.base_seed = seed if seed is not None else int(time.time()) % 10000
        self.current_seed = self.base_seed
        
        # Set up logging
        self.enable_logging = enable_logging and LOGGER_AVAILABLE
        if self.enable_logging:
            log_dir = os.path.join(os.getcwd(), 'logs')
            self.logger = SimulationLogger(log_dir=log_dir)
            self.logger.log_config(CONFIG)
        else:
            self.logger = DummyLogger()
        
        # Reset to initialize
        self.reset()
    
    def reset(self, specific_seed=None):
        self.step_count = 0
        self.task_state = None
        self._prev_distances = None
        
        # Clear existing agents and obstacles
        self.uavs = []
        self.obstacles = []
        
        # Update seed management for deterministic reproducibility if needed
        if specific_seed is not None:
            # Use provided seed for this specific reset
            self.current_seed = specific_seed
        else:
            # Generate a new seed based on current time or increment the existing one
            self.current_seed = (self.base_seed + self.step_count) % 100000
        
        # Set the random seed
        np.random.seed(self.current_seed)
        
        # Log the beginning of this episode with the seed used
        self.logger.log_episode_start(
            episode=self.episode_num,  # Use episode_num here
            seed=self.current_seed,
            positions={"seed": self.current_seed}  # Will be updated later with actual positions
        )
        
        # Define the possible regions - dividing the space into sections
        # This helps ensure better distribution of entities
        regions = [
            (0.1, self.width/2, 0.1, self.height/2),             # Bottom-left
            (self.width/2, self.width-0.1, 0.1, self.height/2),     # Bottom-right
            (0.1, self.width/2, self.height/2, self.height-0.1),    # Top-left
            (self.width/2, self.width-0.1, self.height/2, self.height-0.1)  # Top-right
        ]
        
        # STEP 1: FIRST CREATE OBSTACLES
        # This is critical to ensure UAVs don't start inside or too close to obstacles
        obstacle_pos_log = []
        obstacle_radius_log = []
        
        # First try to distribute obstacles in regions for better coverage
        remaining_obstacles = self.num_obstacles
        obstacles_per_region = max(1, self.num_obstacles // 4)  # Distribute across 4 regions
        
        for region_idx, (x_min, x_max, y_min, y_max) in enumerate(regions):
            if remaining_obstacles <= 0:
                break
                
            # Place obstacles_per_region in each region
            for _ in range(min(obstacles_per_region, remaining_obstacles)):
                attempts = 0
                while attempts < 50:  # Reduced attempts per region
                    attempts += 1
                    
                    # Randomize obstacle position within this region
                    obs_pos = np.array([
                        np.random.uniform(x_min, x_max),
                        np.random.uniform(y_min, y_max)
                    ])
                    
                    # Randomize obstacle size within configured limits
                    obs_radius = np.random.uniform(CONFIG["obstacle_radius_min"], CONFIG["obstacle_radius_max"])
                    
                    # At this point, no UAVs or target exists yet, so we only need to check against existing obstacles
                    # Ensure obstacle doesn't overlap with other obstacles
                    if all(np.linalg.norm(obs_pos - obs.position) > obs_radius + obs.radius + 1.0 for obs in self.obstacles):
                        self.obstacles.append(Obstacle(obs_pos, obs_radius))
                        obstacle_pos_log.append(obs_pos.tolist())
                        obstacle_radius_log.append(float(obs_radius))
                        remaining_obstacles -= 1
                        break
                
                # If we couldn't place an obstacle in this region after max attempts, continue to next region
                if attempts >= 50:
                    continue
        
        # If we still have obstacles to place, try anywhere in the scenario
        for i in range(remaining_obstacles):
            attempts = 0
            while attempts < 100:
                attempts += 1
                
                # Randomize obstacle position
                obs_pos = np.array([
                    np.random.uniform(0.1, self.width - 0.1),
                    np.random.uniform(0.1, self.height - 0.1)
                ])
                
                # Randomize obstacle size within configured limits
                obs_radius = np.random.uniform(CONFIG["obstacle_radius_min"], CONFIG["obstacle_radius_max"])
                
                # Ensure obstacle doesn't overlap with other obstacles with extra margin
                if all(np.linalg.norm(obs_pos - obs.position) > obs_radius + obs.radius + 1.0 for obs in self.obstacles):
                    self.obstacles.append(Obstacle(obs_pos, obs_radius))
                    obstacle_pos_log.append(obs_pos.tolist())
                    obstacle_radius_log.append(float(obs_radius))
                    break
        
        # STEP 2: NOW CREATE UAVs OUTSIDE OBSTACLES
        # With obstacles already in place, we can properly place UAVs outside their detection range
        uav_positions = []
        colors = ['red', 'blue', 'green']  # As shown in the paper
        uav_pos_log = []
        
        # Define constant for safe distance from obstacles (enlarged radius) - make this quite large
        OBSTACLE_SAFETY_MARGIN = 20  # Large safety margin to keep UAVs far from obstacles
        
        for i in range(self.num_uavs):
            # Select a random region for this UAV
            region_idx = i % len(regions) if self.num_uavs <= len(regions) else np.random.randint(0, len(regions))
            x_min, x_max, y_min, y_max = regions[region_idx]
            
            attempts = 0
            max_attempts = 200  # Increased attempt limit for safe placement
            positioned = False
            
            while attempts < max_attempts and not positioned:
                attempts += 1
                
                # Generate potential position within region
                pos = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max)
                ])
                
                # First check minimum distance from other UAVs
                if not all(np.linalg.norm(pos - p) > 3 * CONFIG["uav_radius"] for p in uav_positions):
                    continue  # Too close to other UAVs, try again
                
                # Most importantly, check if position is far enough from ALL obstacles
                safe_from_obstacles = True
                for obstacle in self.obstacles:
                    # Use enlarged radius with large safety margin
                    enlarged_radius = obstacle.radius * OBSTACLE_SAFETY_MARGIN
                    if np.linalg.norm(pos - obstacle.position) < enlarged_radius:
                        safe_from_obstacles = False
                        break
                
                # Only add UAV if it's safe from obstacles
                if safe_from_obstacles:
                    uav_positions.append(pos)
                    self.uavs.append(UAV(i, pos, colors[i % len(colors)]))
                    uav_pos_log.append(pos.tolist())
                    positioned = True
            
            # If we still couldn't find a safe position after many attempts in the current region,
            # try with a reduced safety margin but still ensuring it's outside obstacles
            if not positioned:
                # Try with progressively smaller safety margins
                reduced_margins = [15, 10, 8, 5]
                
                for safety_margin in reduced_margins:
                    attempts = 0
                    while attempts < 50 and not positioned:
                        attempts += 1
                        pos = np.array([
                            np.random.uniform(0.1, self.width - 0.1),
                            np.random.uniform(0.1, self.height - 0.1)
                        ])
                        
                        # Check minimum distance from other UAVs (can be closer in this emergency placement)
                        if not all(np.linalg.norm(pos - p) > 2 * CONFIG["uav_radius"] for p in uav_positions):
                            continue
                        
                        # Check distance from obstacles with reduced margin
                        safe_from_obstacles = True
                        for obstacle in self.obstacles:
                            enlarged_radius = obstacle.radius * safety_margin
                            if np.linalg.norm(pos - obstacle.position) < enlarged_radius:
                                safe_from_obstacles = False
                                break
                        
                        if safe_from_obstacles:
                            uav_positions.append(pos)
                            self.uavs.append(UAV(i, pos, colors[i % len(colors)]))
                            uav_pos_log.append(pos.tolist())
                            self.logger.log_warning(f"UAV {i} placed with reduced safety margin of {safety_margin}")
                            positioned = True
                            break
                
                # If we still can't place it safely, log an error and place it while ensuring it's at least
                # outside the actual obstacle boundary plus a small buffer
                if not positioned:
                    while True:
                        pos = np.array([
                            np.random.uniform(0.1, self.width - 0.1),
                            np.random.uniform(0.1, self.height - 0.1)
                        ])
                        
                        # Ensure position is outside all actual obstacles with a small buffer
                        if all(np.linalg.norm(pos - obs.position) > (obs.radius * 10) + CONFIG["uav_radius"] + 0.5 
                               for obs in self.obstacles):
                            uav_positions.append(pos)
                            self.uavs.append(UAV(i, pos, colors[i % len(colors)]))
                            uav_pos_log.append(pos.tolist())
                            self.logger.log_error(f"UAV {i} placed with minimal safety margin after {max_attempts} attempts")
                            break
        
        # STEP 3: FINALLY CREATE TARGET away from UAVs and obstacles
        target_pos_log = None
        attempts = 0
        max_attempts = 200  # Increased attempts for better placement
        target_created = False
        
        # Curriculum learning: place target closer to UAVs in early episodes
        if self.episode_num < 10:
            min_distance = 5 * CONFIG["uav_radius"]
            self.logger.log_error(f"Curriculum learning: Episode {self.episode_num}, placing target closer to UAVs (min_distance={min_distance})")
        else:
            min_distance = 10 * CONFIG["uav_radius"]
            self.logger.log_error(f"Curriculum learning: Episode {self.episode_num}, placing target at normal distance (min_distance={min_distance})")
        
        while attempts < max_attempts and not target_created:
            attempts += 1
            target_pos = np.array([
                np.random.uniform(0.1, self.width - 0.1),
                np.random.uniform(0.1, self.height - 0.1)
            ])
            
            # Ensure target is not too close to UAVs
            if not all(np.linalg.norm(target_pos - uav.position) > min_distance for uav in self.uavs):
                continue
                
            # Also ensure target is not inside any obstacle or too close to obstacles
            safe_from_obstacles = True
            for obstacle in self.obstacles:
                # Use a reasonable margin for targets around obstacles
                safe_distance = obstacle.radius * 10 + CONFIG["target_radius"] + 1.0
                if np.linalg.norm(target_pos - obstacle.position) < safe_distance:
                    safe_from_obstacles = False
                    break
                    
            if safe_from_obstacles:
                self.target = Target(target_pos)
                target_pos_log = target_pos.tolist()
                target_created = True
        
        # If we still couldn't find a suitable position after all attempts, place it anywhere safe from obstacles
        if not target_created:
            while True:
                target_pos = np.array([
                    np.random.uniform(0.1, self.width - 0.1),
                    np.random.uniform(0.1, self.height - 0.1)
                ])
                
                # Ensure it's at least outside any obstacle:
                if all(np.linalg.norm(target_pos - obs.position) > obs.radius * 10 + CONFIG["target_radius"] + 0.5 
                        for obs in self.obstacles):
                    self.target = Target(target_pos)
                    target_pos_log = target_pos.tolist()
                    self.logger.log_error(f"Target placed with minimal safety margin after {max_attempts} attempts")
                    break
        
        # Log all initial positions
        position_data = {
            "uavs": uav_pos_log,
            "target": target_pos_log,
            "obstacles": {"positions": obstacle_pos_log, "radii": obstacle_radius_log}
        }
        self.logger.log_episode_start(
            episode=self.episode_num,  # Use episode_num here
            seed=self.current_seed,
            positions=position_data
        )
        
        # Update sensors for all agents
        for uav in self.uavs:
            uav.update_sensors(self.obstacles, self.width, self.height)
        
        # Get initial state
        return self.get_state()
    
    def get_state(self):
        # Combine states from all UAVs and target
        state = np.array([], dtype=np.float32)
        
        # Add UAV states
        for uav in self.uavs:
            uav_state = uav.get_state(self.width, self.height)
            state = np.concatenate([state, uav_state])
        
        # Add other UAVs' positions relative to each UAV
        for i, uav in enumerate(self.uavs):
            for j, other_uav in enumerate(self.uavs):
                if i != j:
                    rel_pos = (other_uav.position - uav.position) / np.array([self.width, self.height])
                    state = np.concatenate([state, rel_pos])
        
        # Add target information (distance and relative angle to each UAV)
        for uav in self.uavs:
            # Distance to target (normalized)
            dist_to_target = np.linalg.norm(self.target.position - uav.position) / np.sqrt(self.width**2 + self.height**2)
            
            # Angle to target (normalized to [-1, 1])
            angle_to_target = math.atan2(
                self.target.position[1] - uav.position[1],
                self.target.position[0] - uav.position[0]
            ) / math.pi
            
            state = np.concatenate([state, [dist_to_target, angle_to_target]])
        
        # Add sensor data for all UAVs
        for uav in self.uavs:
            state = np.concatenate([state, uav.get_normalized_sensor_data()])
        
        return state
    
    def get_state_dim(self):
        # Calculate the dimension of the state space
        state = self.get_state()
        return state.shape[0]
    
    def _apply_obstacle_avoidance(self, uav, original_force):
        """Apply obstacle avoidance behavior based on sensor readings - simplified for smoother movement."""
        # Update sensors to get fresh readings
        uav.update_sensors(self.obstacles, self.width, self.height)
        
        # Constants for avoidance behavior - stronger values for more reliable avoidance
        AVOIDANCE_THRESHOLD = CONFIG["sensor_range"] * 3.5  # Start avoiding at 350% of max range - earlier detection
        MAX_AVOIDANCE_FORCE = CONFIG["uav_max_acceleration"] * 3.0  # Stronger maximum force
        EMERGENCY_THRESHOLD = 0.5  # Increased buffer for emergency maneuvers
        
        # Initialize avoidance force
        avoidance_force = np.zeros(2)
        
        # Track closest obstacle and its properties
        min_obstacle_distance = float('inf')
        closest_obstacle = None
        emergency_situation = False
        
        # First check for emergency situations (very close to any obstacle)
        for obstacle in self.obstacles:
            # Calculate distance to obstacle center
            obstacle_distance = np.linalg.norm(uav.position - obstacle.position)
            # Using the enlarged radius (10x) to match visualization
            enlarged_radius = obstacle.radius * 10
            safe_distance = enlarged_radius + CONFIG["uav_radius"]
            
            # Keep track of closest obstacle
            if obstacle_distance < min_obstacle_distance:
                min_obstacle_distance = obstacle_distance
                closest_obstacle = obstacle
                
            # Check for emergency situation (imminent collision)
            if obstacle_distance <= safe_distance + EMERGENCY_THRESHOLD:
                emergency_situation = True
        
        # Handle emergency avoidance if necessary
        if emergency_situation and closest_obstacle:
            # Calculate direction away from obstacle
            escape_dir = uav.position - closest_obstacle.position
            escape_norm = np.linalg.norm(escape_dir)
            
            if escape_norm > 0.001:  # Avoid division by zero
                escape_dir = escape_dir / escape_norm
            else:
                # If exactly at center, choose random direction
                angle = np.random.random() * 2 * np.pi
                escape_dir = np.array([np.cos(angle), np.sin(angle)])
            
            # Calculate a more dynamic emergency force based on how close we are
            # Scale force from MAX_FORCE (at exact collision) to a gentler force at the threshold
            emergency_dist = min_obstacle_distance - (closest_obstacle.radius * 10 + CONFIG["uav_radius"])
            # Normalize distance into a 0-1 range where 0 is collision and 1 is at emergency threshold
            norm_dist = max(0, min(1, emergency_dist / EMERGENCY_THRESHOLD))
            # Apply a stronger curve for force transition
            force_factor = 8.0 * (1.0 - norm_dist**2)  # Enhanced quadratic falloff for stronger avoidance
            emergency_force = escape_dir * CONFIG["uav_max_acceleration"] * CONFIG["uav_mass"] * force_factor
            
            # Blend with previous force for ultra-smooth emergency response, but prioritize new force more
            if hasattr(uav, 'prev_avoidance_force') and np.linalg.norm(uav.prev_avoidance_force) > 0.001:
                # More aggressive blending for emergencies but still with some smoothing
                blend_factor = 0.15  # 85% new force, 15% old force - more responsive
                emergency_force = (1.0 - blend_factor) * emergency_force + blend_factor * uav.prev_avoidance_force
            
            # Update previous force for next iteration
            uav.prev_avoidance_force = emergency_force.copy()
            return emergency_force
        
        # Process sensor data for normal avoidance behavior
        # Process all sensors to create a more holistic avoidance response
        # This accounts for multiple obstacles from different directions
        weighted_directions = np.zeros(2)
        total_weight = 0
        
        for i, distance in enumerate(uav.sensor_data):
            if distance < AVOIDANCE_THRESHOLD:
                # Calculate angle for this sensor
                angle = 2 * math.pi * i / CONFIG["num_sensors"]
                
                # Calculate proximity factor (1.0 when very close, 0.0 at threshold distance)
                proximity_factor = max(0, 1.0 - (distance / AVOIDANCE_THRESHOLD))
                
                # Apply a smoother response curve
                # Cubic curve gives very gentle initial response and stronger close response
                smoothed_factor = proximity_factor ** 2.2  # More gradual curve
                
                # Basic avoidance direction (away from obstacle)
                avoidance_dir = np.array([-math.cos(angle), -math.sin(angle)])
                
                # Dynamic tangential component based on UAV's current velocity
                # This helps UAVs navigate around obstacles more naturally
                vel_magnitude = np.linalg.norm(uav.velocity)
                
                # Choose tangent direction that aligns better with current velocity when possible
                # This helps prevent the UAV from zigzagging around obstacles
                tangent_angle_1 = angle + math.pi/2  # Clockwise tangent
                tangent_angle_2 = angle - math.pi/2  # Counter-clockwise tangent
                tangent_dir_1 = np.array([math.cos(tangent_angle_1), math.sin(tangent_angle_1)])
                tangent_dir_2 = np.array([math.cos(tangent_angle_2), math.sin(tangent_angle_2)])
                
                # Prefer the tangent direction that's more aligned with current velocity
                if vel_magnitude > 0.1:
                    vel_norm = uav.velocity / vel_magnitude
                    dot1 = np.dot(vel_norm, tangent_dir_1)
                    dot2 = np.dot(vel_norm, tangent_dir_2)
                    tangent_dir = tangent_dir_1 if dot1 > dot2 else tangent_dir_2
                else:
                    # Default to clockwise if velocity is too low
                    tangent_dir = tangent_dir_1
                
                # Dynamic blending ratio based on proximity
                # Closer = more direct avoidance, further = more tangential movement
                avoidance_ratio = 0.6 + 0.3 * smoothed_factor  # 0.6-0.9 range
                tangent_ratio = 1.0 - avoidance_ratio
                
                # Blend directions with dynamic ratio
                blended_dir = avoidance_ratio * avoidance_dir + tangent_ratio * tangent_dir
                blended_dir = blended_dir / np.linalg.norm(blended_dir)  # Normalize
                
                # Apply weight based on proximity (closer obstacles have more influence)
                weight = smoothed_factor * smoothed_factor  # Square for more emphasis on close obstacles
                weighted_directions += blended_dir * weight
                total_weight += weight
        
        # If obstacles detected, calculate the final avoidance force
        if total_weight > 0.001:
            # Normalize the combined direction
            avoidance_dir = weighted_directions / total_weight
            avoidance_dir = avoidance_dir / np.linalg.norm(avoidance_dir)  # Ensure unit length
            
            # Calculate overall proximity factor based on the closest obstacle
            # This controls the overall magnitude of the avoidance response
            closest_sensor_dist = min(uav.sensor_data)
            if closest_sensor_dist < AVOIDANCE_THRESHOLD:
                overall_proximity = max(0, 1.0 - (closest_sensor_dist / AVOIDANCE_THRESHOLD))
                overall_smoothed = overall_proximity ** 1.8  # Smoother overall response curve
                
                # Calculate avoidance force with smooth scaling
                avoidance_force = avoidance_dir * MAX_AVOIDANCE_FORCE * overall_smoothed
                
                # Continuous blending weight based on proximity instead of discrete steps
                # Improved sigmoid-like function for weight transition that prioritizes avoidance more
                avoidance_weight = 0.4 + 0.6 * (1 / (1 + math.exp(-12 * (overall_proximity - 0.4))))
                
                # Blend original force with avoidance force
                combined_force = original_force * (1 - avoidance_weight) + avoidance_force * avoidance_weight
                
                # Apply temporal smoothing with the previous force
                # This is crucial for eliminating jerky movements
                if hasattr(uav, 'prev_avoidance_force'):
                    # Less aggressive smoothing for more responsive movement to obstacles
                    # Adjust these values for more/less smoothing as needed
                    smooth_factor = 0.25  # Lower = less smoothing (0.25 = 75% new, 25% old)
                    combined_force = (1.0 - smooth_factor) * combined_force + smooth_factor * uav.prev_avoidance_force
                
                # Store for next iteration
                uav.prev_avoidance_force = combined_force.copy()
                
                # Apply force limiting to prevent excessive acceleration
                # Use a soft limit to prevent sudden changes in force magnitude
                max_force = CONFIG["uav_max_acceleration"] * CONFIG["uav_mass"]
                force_magnitude = np.linalg.norm(combined_force)
                if force_magnitude > max_force:
                    # Scale down while preserving direction
                    scaling_factor = max_force / force_magnitude
                    combined_force = combined_force * scaling_factor
                
                return combined_force
        
        # Gradual transition to original force when no obstacles detected
        # This helps prevent sudden changes when obstacles go in/out of range
        if hasattr(uav, 'prev_avoidance_force') and np.linalg.norm(uav.prev_avoidance_force) > 0.001:
            # Very gentle transition back to original force
            transition_weight = 0.2  # 80% new, 20% old
            blended_force = (1.0 - transition_weight) * original_force + transition_weight * uav.prev_avoidance_force
            uav.prev_avoidance_force = blended_force.copy()
            return blended_force
        else:
            # No previous force or zero previous force, use original
            uav.prev_avoidance_force = original_force.copy()
            return original_force
    
    def step(self, actions):
        self.step_count += 1
        
        # Save distances for reward calculation
        current_distances = [np.linalg.norm(uav.position - self.target.position) for uav in self.uavs]
        self._prev_distances = current_distances.copy()
        
        # Track positions and collisions for logging
        uav_positions = []
        collision_events = []
        
        # Apply actions (forces) to UAVs
        for i, uav in enumerate(self.uavs):
            # Save previous position before update
            prev_position = uav.position.copy()
            
            # Get base force from original action
            original_force = np.array(actions[i]) * CONFIG["uav_max_acceleration"] * CONFIG["uav_mass"]
            
            # Apply obstacle avoidance to get modified force
            modified_force = self._apply_obstacle_avoidance(uav, original_force)
            
            # Update UAV with the modified force
            uav.update(modified_force)
            
            # Track position for logging
            uav_positions.append(uav.position.copy().tolist())
            
            # Ensure UAVs stay within boundaries with absolute enforcement
            # Check if we're at a boundary before clipping
            at_boundary = False
            if (uav.position[0] < CONFIG["uav_radius"] or 
                uav.position[0] > self.width - CONFIG["uav_radius"] or
                uav.position[1] < CONFIG["uav_radius"] or
                uav.position[1] > self.height - CONFIG["uav_radius"]):
                at_boundary = True
                collision_events.append({"type": "boundary", "uav_id": i})
            
            # Clip positions to boundaries
            uav.position[0] = np.clip(uav.position[0], 0 + CONFIG["uav_radius"], self.width - CONFIG["uav_radius"])
            uav.position[1] = np.clip(uav.position[1], 0 + CONFIG["uav_radius"], self.height - CONFIG["uav_radius"])
            
            # If UAV is at the boundary edge, reflect its velocity to prevent it from trying to exit
            if at_boundary:
                # Left boundary
                if uav.position[0] <= CONFIG["uav_radius"] + 0.1:
                    uav.velocity[0] = abs(uav.velocity[0]) * 0.8  # Reflect and dampen
                # Right boundary
                elif uav.position[0] >= self.width - CONFIG["uav_radius"] - 0.1:
                    uav.velocity[0] = -abs(uav.velocity[0]) * 0.8  # Reflect and dampen
                # Bottom boundary
                if uav.position[1] <= CONFIG["uav_radius"] + 0.1:
                    uav.velocity[1] = abs(uav.velocity[1]) * 0.8  # Reflect and dampen
                # Top boundary
                elif uav.position[1] >= self.height - CONFIG["uav_radius"] - 0.1:
                    uav.velocity[1] = -abs(uav.velocity[1]) * 0.8  # Reflect and dampen
            
            # Post-movement collision check and resolution
            # This is a safety measure to absolutely prevent UAVs from passing through obstacles
            for obs_idx, obstacle in enumerate(self.obstacles):
                # Distance between UAV center and obstacle center
                distance = np.linalg.norm(uav.position - obstacle.position)
                # Using the enlarged obstacle radius (10x) to match visualization
                enlarged_radius = obstacle.radius * 10
                safe_distance = enlarged_radius + CONFIG["uav_radius"]
                
                # If UAV is inside obstacle, move it outside
                if distance < safe_distance:
                    # Log collision event
                    collision_events.append({
                        "type": "obstacle", 
                        "uav_id": i, 
                        "obstacle_id": obs_idx,
                        "distance": float(distance)
                    })
                    
                    # Calculate direction from obstacle to UAV
                    direction = uav.position - obstacle.position
                    direction_norm = np.linalg.norm(direction)
                    
                    if direction_norm > 0.001:  # Avoid division by zero
                        # Normalize direction vector
                        direction = direction / direction_norm
                        # Move UAV to the edge of the obstacle plus a small buffer
                        uav.position = obstacle.position + direction * (safe_distance + 0.05)
                        # Reflect velocity away from obstacle to prevent getting stuck
                        reflection = uav.velocity - 2 * np.dot(uav.velocity, direction) * direction
                        uav.velocity = reflection * 0.5  # Reduce velocity to 50% after collision
                    else:
                        # If UAV is exactly at obstacle center (very unlikely), move in random direction
                        angle = np.random.random() * 2 * np.pi
                        random_dir = np.array([np.cos(angle), np.sin(angle)])
                        uav.position = obstacle.position + random_dir * (safe_distance + 0.05)
                        uav.velocity = random_dir * 0.5 * CONFIG["uav_max_velocity"]
        
        # Ensure target always stays within boundaries regardless of dynamic_obstacles setting
        # We'll use a small buffer distance from the edge
        buffer = CONFIG["uav_radius"]
        target_prev_pos = self.target.position.copy()
        
        # Target handling - with boundary avoidance and sensors
        if self.dynamic_obstacles:
            # Target takes evasive action only when dynamic_obstacles is enabled
            target_force = self._get_target_evasive_action()
            # Use the improved update method with boundary sensors
            # Pass scenario dimensions explicitly to ensure correct boundary detection
            self.target.update(target_force, scenario_width=self.width, scenario_height=self.height)
        else:
            # When dynamic_obstacles is disabled, target remains static
            # But still need to update trajectory for visualization consistency
            if hasattr(self.target, 'trajectory'):
                self.target.trajectory.append(self.target.position.copy())
        
        # Check if target hit boundary
        target_at_boundary = False
        if (self.target.position[0] < buffer or 
            self.target.position[0] > self.width - buffer or
            self.target.position[1] < buffer or
            self.target.position[1] > self.height - buffer):
            target_at_boundary = True
            collision_events.append({"type": "target_boundary"})
        
        # Final safety check - clip target position to boundaries with buffer
        # This is just a backup; the target's internal boundary avoidance should prevent
        # it from ever getting this close to the edge in normal operation
        self.target.position[0] = np.clip(self.target.position[0], 0 + buffer, self.width - buffer)
        self.target.position[1] = np.clip(self.target.position[1], 0 + buffer, self.height - buffer)
        
        # Track obstacle movements for logging
        obstacle_movements = []
        
        # Update dynamic obstacles if enabled
        if self.dynamic_obstacles:
            for obs_idx, obstacle in enumerate(self.obstacles):
                prev_pos = obstacle.position.copy()
                
                # Random movement for obstacles
                if np.random.random() < 0.1:  # 10% chance to change direction
                    obstacle.velocity = np.random.uniform(-0.02, 0.02, size=2)
                
                obstacle.position += obstacle.velocity
                
                # Check if obstacle hit boundary
                obs_at_boundary = False
                if (obstacle.position[0] < obstacle.radius or 
                    obstacle.position[0] > self.width - obstacle.radius or
                    obstacle.position[1] < obstacle.radius or
                    obstacle.position[1] > self.height - obstacle.radius):
                    obs_at_boundary = True
                    collision_events.append({
                        "type": "obstacle_boundary", 
                        "obstacle_id": obs_idx
                    })
                
                # Keep obstacles within boundaries
                obstacle.position[0] = np.clip(obstacle.position[0], obstacle.radius, self.width - obstacle.radius)
                obstacle.position[1] = np.clip(obstacle.position[1], obstacle.radius, self.height - obstacle.radius)
                
                # Track obstacle movement
                if not np.array_equal(prev_pos, obstacle.position):
                    obstacle_movements.append({
                        "obstacle_id": obs_idx,
                        "prev_pos": prev_pos.tolist(),
                        "new_pos": obstacle.position.tolist(),
                        "hit_boundary": obs_at_boundary
                    })
        
        # Update sensors
        for uav in self.uavs:
            uav.update_sensors(self.obstacles, self.width, self.height)
        
        # Calculate rewards and check terminal conditions
        rewards, done, info = self._calculate_rewards()
        
        # Get new state
        next_state = self.get_state()
        
        # Log step information
        step_log = {
            "uav_positions": uav_positions,
            "target_position": self.target.position.tolist(),
            "collision_events": collision_events,
            "obstacle_movements": obstacle_movements,
            "rewards": rewards,
            "distances": [np.linalg.norm(uav.position - self.target.position) for uav in self.uavs],
            "status": info,
            "done": done
        }
        
        # Only log full details occasionally to reduce file size, or when significant events occur
        if self.step_count % 10 == 0 or done or len(collision_events) > 0 or info['success']:
            self.logger.log_step(self.step_count, "episode", actions, next_state, rewards)
        
        # Log if target is captured
        if info['success'] and not done:
            self.logger.log_capture("episode", self.step_count)
        
        # Log episode completion
        if done:
            self.logger.log_episode_end("episode", sum(rewards), info['success'], self.step_count)
            self.episode_num += 1  # Increment episode_num here
        
        return next_state, rewards, done, info
    
    def _get_target_evasive_action(self):
        """Simplified evasion strategy: move away from UAVs and maintain maximum speed."""
        # Calculate direction vector away from UAVs
        direction = np.zeros(2)
        for uav in self.uavs:
            dist_vec = self.target.position - uav.position
            dist = np.linalg.norm(dist_vec)
            
            # Weight by inverse square of distance
            if dist > 0:
                direction += dist_vec / (dist**2)
        
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        else:
            # If direction is zero, choose a random direction
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Apply maximum acceleration in that direction
        return direction * CONFIG["target_max_acceleration"]
    
    def _calculate_rewards(self):
        rewards = []
        done = False
        info = {
            "task_state": None,
            "success": False
        }
        
        # Calculate distances from UAVs to target
        current_distances = [np.linalg.norm(uav.position - self.target.position) for uav in self.uavs]
        
        # Calculate triangular areas for encirclement detection
        triangle_areas = []
        total_encirclement_area = 0
        
        if self.num_uavs >= 3:  # Need at least 3 UAVs to form an encirclement
            for i in range(self.num_uavs):
                next_i = (i + 1) % self.num_uavs
                # Calculate area of triangle formed by two UAVs and the target
                area = calculate_triangle_area(
                    self.uavs[i].position,
                    self.uavs[next_i].position,
                    self.target.position
                )
                triangle_areas.append(area)
            
            # Sum of areas is the total encirclement area
            total_encirclement_area = sum(triangle_areas)
        
        # Check if target is within the convex hull formed by UAVs
        encirclement_formed = self._is_target_encircled()
        
        # Determine task state based on conditions from the paper
        total_distance = sum(current_distances)
        all_captured = all(d <= CONFIG["capture_distance"] for d in current_distances)
        any_captured = any(d <= CONFIG["capture_distance"] for d in current_distances)
        
        # Determine task state according to paper's criteria
        if not encirclement_formed and total_distance >= CONFIG["d_limit"] and not any_captured:
            task_state = "tracking"
        elif not encirclement_formed and (total_distance < CONFIG["d_limit"] or any_captured):
            task_state = "encircling"
        elif encirclement_formed and not all_captured:
            task_state = "capturing"
        else:
            task_state = "finished"
            done = True
            info["success"] = True
        
        info["task_state"] = task_state
        self.task_state = task_state
        
        # Calculate rewards for each UAV based on the curriculum
        for i, uav in enumerate(self.uavs):
            # Base rewards common to all tasks
            approach_reward = self._calculate_approach_reward(uav)
            safety_reward = self._calculate_safety_reward(uav)
            
            # Task-specific reward based on curriculum learning
            task_reward = 0
            
            if task_state == "tracking":
                # Tracking reward: negative sum of distances (encourage getting closer)
                task_reward = -total_distance / (np.sqrt(self.width**2 + self.height**2) * self.num_uavs)
                
            elif task_state == "encircling":
                # Encircling reward: based on formation of encirclement
                if self.num_uavs >= 3:
                    # The paper uses ln(sum of triangles - total encirclement + 1)
                    encircle_diff = abs(sum(triangle_areas) - total_encirclement_area)
                    task_reward = -np.log(encircle_diff + 1) / self.num_uavs
                
            elif task_state == "capturing":
                # Capturing reward: based on shrinking the encirclement
                if self._prev_distances is not None:
                    # Positive reward if getting closer to target
                    distance_change = sum(self._prev_distances) - sum(current_distances)
                    task_reward = np.exp(distance_change / (self.num_uavs * CONFIG["uav_max_velocity"]))
            
            # Finish reward if task completed
            finish_reward = 10.0 if task_state == "finished" else 0.0
            
            # Combine rewards with appropriate weights
            if self.episode_num < 10:
                total_reward = (
                    CONFIG["reward_weights"]["approach"] * approach_reward +
                    CONFIG["reward_weights"]["safety"] * safety_reward +
                    CONFIG["reward_weights"]["finish"] * finish_reward +
                    CONFIG["reward_weights"]["track"] * task_reward
                )
            else:
                total_reward = (
                    CONFIG["reward_weights"]["approach"] * approach_reward +
                    CONFIG["reward_weights"]["safety"] * safety_reward +
                    CONFIG["reward_weights"]["finish"] * finish_reward +
                    CONFIG["reward_weights"]["encircle"] * task_reward
                )
            
            rewards.append(total_reward)
        
        # Check for timeout
        if self.step_count >= CONFIG["max_steps_per_episode"]:
            done = True
        
        return rewards, done, info
    
    def _calculate_approach_reward(self, uav):
        """Reward for approaching the target, as per equation (18) in the paper."""
        vel_norm = np.linalg.norm(uav.velocity)
        if vel_norm < 1e-6:
            return 0
        
        # Calculate relative azimuth, as per equation (19)
        target_angle = math.atan2(
            self.target.position[1] - uav.position[1],
            self.target.position[0] - uav.position[0]
        )
        
        velocity_angle = math.atan2(uav.velocity[1], uav.velocity[0])
        rel_azimuth = velocity_angle - target_angle
        
        # Normalize velocity and calculate reward
        normalized_vel = vel_norm / CONFIG["uav_max_velocity"]
        approach_reward = normalized_vel * math.cos(rel_azimuth)
        
        return approach_reward
    
    def _calculate_safety_reward(self, uav):
        """Safety reward for avoiding obstacles and boundaries, as per equation (20)."""
        # Check collision with obstacles
        for obstacle in self.obstacles:
            dist_to_obstacle = np.linalg.norm(uav.position - obstacle.position) - obstacle.radius
            if dist_to_obstacle <= 0:
                return -10.0  # Large penalty for collision
        
        # Check if out of bounds
        if (uav.position[0] < 0 or uav.position[0] > self.width or
            uav.position[1] < 0 or uav.position[1] > self.height):
            return -10.0  # Large penalty for going out of bounds
        
        # Penalize being close to obstacles or boundaries based on sensor readings
        min_sensor = min(uav.sensor_data)
        safety_reward = (min_sensor - CONFIG["sensor_range"]) / CONFIG["sensor_range"]
        
        return safety_reward
    
    def _is_target_encircled(self):
        """Check if the target is inside the convex hull formed by UAVs."""
        if self.num_uavs < 3:
            return False
        
        # Get UAV positions
        uav_positions = [uav.position for uav in self.uavs]
        target_pos = self.target.position
        
        # Check if target is inside the convex hull using ray casting algorithm
        # This is a simplified version and can be improved for more accurate detection
        in_polygon = False
        
        for i in range(self.num_uavs):
            j = (i + 1) % self.num_uavs
            
            if ((uav_positions[i][1] > target_pos[1]) != (uav_positions[j][1] > target_pos[1]) and
                (target_pos[0] < (uav_positions[j][0] - uav_positions[i][0]) * 
                 (target_pos[1] - uav_positions[i][1]) / 
                 (uav_positions[j][1] - uav_positions[i][1]) + uav_positions[i][0])):
                in_polygon = not in_polygon
        
        return in_polygon