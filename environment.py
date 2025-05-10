# Multi-UAV Simulation Environment
# Implements the physics, obstacle avoidance, and UAV coordination mechanics
# Enhanced with improved obstacle avoidance and visualization

import numpy as np
import math
import time
import os
import sys
from datetime import datetime
from utils import CONFIG, calculate_triangle_area

from agents import UAV, Target, Obstacle, AHFSI_AVAILABLE

# Try to import AHFSI framework components if available
try:
    # Import each component separately to locate the specific import error
    try:
        from ahfsi_framework import AHFSIController
        print("Successfully imported AHFSIController")
    except ImportError as e1:
        print(f"Failed to import AHFSIController: {e1}")
        
    try:
        from swarm_intelligence import SwarmBehavior
        print("Successfully imported SwarmBehavior")
    except ImportError as e2:
        print(f"Failed to import SwarmBehavior: {e2}")
        
    try:
        from information_theory import InformationTheory
        print("Successfully imported InformationTheory")
    except ImportError as e3:
        print(f"Failed to import InformationTheory: {e3}")
    
    # If we've made it this far without raising an exception, all components are available
    AHFSI_COMPONENTS_AVAILABLE = True
except Exception as e:
    AHFSI_COMPONENTS_AVAILABLE = False
    print(f"Warning: AHFSI components not available. Error: {e}. Running with standard behavior.")

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
    """Simulation environment for multiple UAVs with obstacle avoidance.
    
    This environment manages the interactions between UAVs, target, and obstacles.
    It implements the physics simulation, sensor models, obstacle detection and
    avoidance behaviors, and reward calculations for reinforcement learning.
    
    Recent enhancements include:
    - Improved obstacle placement to prevent overlapping
    - Enhanced obstacle avoidance with increased sensitivity
    - Consistent physics and visualization (obstacles appear same size in both)
    - Obstacles are 10x larger visually for better visibility
    """
    def __init__(self, num_uavs=3, num_obstacles=3, dynamic_obstacles=False, seed=None, enable_logging=True, mode="train", enable_ahfsi=True):
        """Initialize the simulation environment.
        
        Args:
            num_uavs: Number of UAVs in the simulation
            num_obstacles: Number of obstacles in the environment
            dynamic_obstacles: Whether obstacles can move (not fully implemented)
            seed: Random seed for reproducibility
            enable_logging: Whether to log simulation data
            enable_ahfsi: Whether to enable AHFSI framework integration
        """
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
        
        # AHFSI framework integration
        self.enable_ahfsi = enable_ahfsi and AHFSI_AVAILABLE and CONFIG["swarm"]["rl_integration"]["enabled"]
        
        # Initialize information theory component for environment-level analysis
        self.information_theory = None
        if AHFSI_COMPONENTS_AVAILABLE and self.enable_ahfsi:
            self.information_theory = InformationTheory()
            
        # Initialize swarm behavior component for coordinated movements
        self.swarm_behavior = None
        if AHFSI_COMPONENTS_AVAILABLE and self.enable_ahfsi:
            self.swarm_behavior = SwarmBehavior()
        
        # Mode determines behavior (train, test, or demo)
        self.mode = "train"
        
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
        
        # Reset information theory component if enabled
        if self.information_theory is not None:
            self.information_theory = InformationTheory()
        
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
        
        # We'll log the episode start after all entities are properly placed
        
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
                    # Using actual physical radius values - obstacles are now physically larger
                    # This ensures obstacles have appropriate physical separation
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
                
                # Use more controlled obstacle sizing to prevent too many large obstacles
                # Limit the first few obstacles to be smaller to ensure UAVs can be placed
                if i < min(2, self.num_obstacles-1):
                    # First obstacles are smaller to ensure placement space for UAVs
                    obs_radius = np.random.uniform(CONFIG["obstacle_radius_min"], 
                                               CONFIG["obstacle_radius_min"] + (CONFIG["obstacle_radius_max"] - CONFIG["obstacle_radius_min"]) * 0.3)
                else:
                    # Later obstacles can be larger
                    obs_radius = np.random.uniform(CONFIG["obstacle_radius_min"], CONFIG["obstacle_radius_max"])
                
                # Ensure obstacle doesn't overlap with other obstacles with extra margin
                # Using actual physical radius values - obstacles are now physically larger
                # This ensures obstacles have appropriate physical separation
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
        
        # Define constant for safe distance from obstacles - use a more reasonable value
        OBSTACLE_SAFETY_MARGIN = 5  # Reduced safety margin to allow for easier UAV placement
        
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
                    self.uavs.append(UAV(i, pos, colors[i % len(colors)], enable_ahfsi=self.enable_ahfsi))
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
                            self.uavs.append(UAV(i, pos, colors[i % len(colors)], enable_ahfsi=self.enable_ahfsi))
                            uav_pos_log.append(pos.tolist())
                            self.logger.log_warning(f"UAV {i} placed with reduced safety margin of {safety_margin}")
                            positioned = True
                            break
                
                # If we still can't place it safely, log an error and place it while ensuring it's at least
                # outside the actual obstacle boundary plus a small buffer
                if not positioned:
                    # Add a limit to prevent infinite loop
                    final_attempts = 0
                    max_final_attempts = 300  # Limit the final placement attempts
                    
                    while final_attempts < max_final_attempts:
                        final_attempts += 1
                        pos = np.array([
                            np.random.uniform(0.1, self.width - 0.1),
                            np.random.uniform(0.1, self.height - 0.1)
                        ])
                        
                        # Ensure position is outside all actual obstacles with a small buffer
                        if all(np.linalg.norm(pos - obs.position) > (obs.radius * 10) + CONFIG["uav_radius"] + 0.5 
                               for obs in self.obstacles):
                            uav_positions.append(pos)
                            self.uavs.append(UAV(i, pos, colors[i % len(colors)], enable_ahfsi=self.enable_ahfsi))
                            uav_pos_log.append(pos.tolist())
                            self.logger.log_error(f"UAV {i} placed with minimal safety margin after {final_attempts} attempts")
                            positioned = True
                            break
                    
                    # If we still couldn't place the UAV after all attempts, force placement and warn user
                    if not positioned:
                        # Force UAV placement in different positions around the environment as last resort
                        # Use different positions for each UAV to avoid overlap
                        corner_positions = [
                            np.array([1.0, 1.0]),  # Bottom left
                            np.array([self.width-1.0, 1.0]),  # Bottom right
                            np.array([1.0, self.height-1.0]),  # Top left
                            np.array([self.width-1.0, self.height-1.0]),  # Top right
                            np.array([self.width/2, self.height/2])  # Center (last resort)
                        ]
                        
                        # Get position based on UAV index to ensure they're distributed
                        corner_pos = corner_positions[i % len(corner_positions)]
                        
                        # Check if this position is clear of obstacles
                        min_distance_to_obstacle = min([np.linalg.norm(corner_pos - obs.position) - obs.radius 
                                                    for obs in self.obstacles], default=float('inf'))
                        
                        if min_distance_to_obstacle < 0.5 and len(self.obstacles) > 0:  # Only if obstacles exist
                            # If our corner position is too close to an obstacle, try to find the clearest area
                            best_pos = None
                            max_distance = -1
                            
                            for test_pos in [
                                np.array([1.0, 1.0]), np.array([self.width-1.0, 1.0]),
                                np.array([1.0, self.height-1.0]), np.array([self.width-1.0, self.height-1.0]),
                                np.array([self.width/2, 1.0]), np.array([1.0, self.height/2]),
                                np.array([self.width-1.0, self.height/2]), np.array([self.width/2, self.height-1.0]),
                                np.array([self.width/2, self.height/2])
                            ]:
                                dist = min([np.linalg.norm(test_pos - obs.position) - obs.radius 
                                           for obs in self.obstacles], default=float('inf'))
                                if dist > max_distance:
                                    max_distance = dist
                                    best_pos = test_pos
                            
                            if best_pos is not None:
                                corner_pos = best_pos
                        
                        uav_positions.append(corner_pos)
                        self.uavs.append(UAV(i, corner_pos, colors[i % len(colors)], enable_ahfsi=self.enable_ahfsi))
                        uav_pos_log.append(corner_pos.tolist())
                        self.logger.log_warning(f"Placed UAV {i} at alternate position {corner_pos} after {max_final_attempts} attempts")
        
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
                safe_distance = obstacle.radius + CONFIG["target_radius"] + 1.0
                if np.linalg.norm(target_pos - obstacle.position) < safe_distance:
                    safe_from_obstacles = False
                    break
                    
            if safe_from_obstacles:
                self.target = Target(target_pos)
                target_pos_log = target_pos.tolist()
                target_created = True
        
        # If we still couldn't find a suitable position after all attempts, place it anywhere safe from obstacles
        if not target_created:
            # Add a limit to prevent infinite loop
            final_attempts = 0
            max_final_attempts = 300  # Limit the final placement attempts
            
            while final_attempts < max_final_attempts and not target_created:
                final_attempts += 1
                target_pos = np.array([
                    np.random.uniform(0.1, self.width - 0.1),
                    np.random.uniform(0.1, self.height - 0.1)
                ])
                
                # Ensure it's at least outside any obstacle:
                if all(np.linalg.norm(target_pos - obs.position) > obs.radius * 10 + CONFIG["target_radius"] + 0.5 
                        for obs in self.obstacles):
                    self.target = Target(target_pos)
                    target_pos_log = target_pos.tolist()
                    self.logger.log_error(f"Target placed with minimal safety margin after {final_attempts} attempts")
                    target_created = True
                    break
                    
            # If we STILL couldn't place the target after all attempts, force placement in center
            if not target_created:
                # Force target placement in center of environment as last resort
                center_pos = np.array([self.width/2, self.height/2])
                self.target = Target(center_pos)
                target_pos_log = center_pos.tolist()
                self.logger.log_error(f"FORCED TARGET placement in center after {max_final_attempts} failed attempts")
        
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
        """Apply obstacle avoidance behavior based on sensor readings.
        
        Implements a sophisticated avoidance system with several key features:
        1. Proactive avoidance starting at 5x sensor range for early path adjustment
        2. Emergency collision prevention for imminent collisions
        3. Smooth blending between avoidance forces for natural movement
        4. Dynamic tangential movement to navigate around obstacles
        5. Multiple obstacle awareness through weighted sensor inputs
        
        Args:
            uav: The UAV to apply avoidance forces to
            original_force: The original control force before avoidance
            
        Returns:
            Modified force vector with obstacle avoidance applied
        """
        # Update sensors to get fresh readings before calculating avoidance
        uav.update_sensors(self.obstacles, self.width, self.height)
        
        # Enhanced avoidance parameters - significantly increased for more effective obstacle avoidance
        # These increased values are a key improvement to the simulation
        AVOIDANCE_THRESHOLD = CONFIG["sensor_range"] * 5.0  # Start avoiding at 500% of max range for much earlier path planning
        MAX_AVOIDANCE_FORCE = CONFIG["uav_max_acceleration"] * 4.5  # 450% stronger force for decisive avoidance
        EMERGENCY_THRESHOLD = 0.8  # Increased safety margin for emergency maneuvers
        
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
            # Using the actual physical radius without visual scaling
            # This provides more realistic physics simulation and avoids model training issues
            # Obstacles are now physically larger as defined in CONFIG
            safe_distance = obstacle.radius + CONFIG["uav_radius"]
            
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
            emergency_dist = min_obstacle_distance - (closest_obstacle.radius + CONFIG["uav_radius"])
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
        
        # Process sensor data for normal avoidance behavior when not in emergency
        # Consider ALL sensors (360-degree awareness) for holistic response
        # This allows detection and avoidance of multiple obstacles simultaneously
        weighted_directions = np.zeros(2)
        total_weight = 0
        
        for i, distance in enumerate(uav.sensor_data):
            if distance < AVOIDANCE_THRESHOLD:
                # Calculate angle for this sensor
                angle = 2 * math.pi * i / CONFIG["num_sensors"]
                
                # Calculate proximity factor (1.0 when very close, 0.0 at threshold distance)
                proximity_factor = max(0, 1.0 - (distance / AVOIDANCE_THRESHOLD))
                
                # Apply a smoother response curve for more natural avoidance
                # Power curve (exponent 2.2) creates gentle initial response far from obstacles
                # and increasingly stronger response as UAV gets closer
                smoothed_factor = proximity_factor ** 2.2  # Optimized gradual curve
                
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
                
                # Dynamic blending ratio based on proximity to obstacle
                # Balance between direct avoidance (moving away) and tangential movement (moving around)
                # Lower avoidance_ratio (0.3-0.6) favors smooth curved paths around obstacles
                # This ratio has been carefully optimized for natural movement patterns
                avoidance_ratio = 0.3 + 0.3 * smoothed_factor  # 0.3-0.6 range 
                # Higher tangent_ratio prioritizes flowing around obstacles instead of backing away
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
                
                # Calculate avoidance force with smooth scaling - ENHANCED scaling for better distance handling
                avoidance_force = avoidance_dir * MAX_AVOIDANCE_FORCE * (overall_smoothed ** 0.8)  # Flatter power curve (0.8) maintains stronger forces at medium distances
                # This power scaling (0.8) ensures UAVs maintain significant avoidance even at greater distances from obstacles
                
                # Continuous blending weight based on proximity instead of discrete steps
                # Improved sigmoid-like function with higher baseline and steeper curve to prioritize avoidance more
                avoidance_weight = 0.6 + 0.4 * (1 / (1 + math.exp(-15 * (overall_proximity - 0.35))))  # Starts higher (0.6), transitions faster
                
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
        
        # Update information theory grid if enabled
        if self.information_theory is not None:
            uav_positions = [uav.position for uav in self.uavs]
            target_position = self.target.position if self.target is not None else None
            self.information_theory.update_entropy_grid(uav_positions, target_position)
        
        # Share information between UAVs if AHFSI is enabled
        if self.enable_ahfsi:
            self._share_swarm_information()
        
        # Apply actions to UAVs
        for i, uav in enumerate(self.uavs):
            if i < len(actions):
                # Convert action from normalized [-1, 1] to force vector
                force = actions[i] * CONFIG["uav_max_acceleration"]
                
                # Apply obstacle avoidance behavior
                force = self._apply_obstacle_avoidance(uav, force)
                
                # Process through AHFSI framework if enabled
                if self.enable_ahfsi:
                    # Create environment state for AHFSI
                    env_state = self._create_ahfsi_state(i)
                    
                    # Get nearby UAVs for this UAV
                    nearby_uavs = self._get_nearby_uavs(i)
                    
                    # Process action through AHFSI framework
                    force = uav.process_rl_action(force, nearby_uavs, env_state)
                
                # Update UAV state
                uav.update(force)
        
        # Track positions and collisions for logging
        uav_positions = []
        collision_events = []
        
        # Ensure UAVs stay within boundaries with absolute enforcement
        # Check if we're at a boundary before clipping
        for uav in self.uavs:
            at_boundary = False
            if (uav.position[0] < CONFIG["uav_radius"] or 
                uav.position[0] > self.width - CONFIG["uav_radius"] or
                uav.position[1] < CONFIG["uav_radius"] or
                uav.position[1] > self.height - CONFIG["uav_radius"]):
                at_boundary = True
                collision_events.append({"type": "boundary", "uav_id": uav.id})
            
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
                # Using actual physical obstacle radius without artificial enlargement
                safe_distance = obstacle.radius + CONFIG["uav_radius"]
                
                # If UAV is inside obstacle, move it outside
                if distance < safe_distance:
                    # Log collision event
                    collision_events.append({
                        "type": "obstacle", 
                        "uav_id": uav.id, 
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
            
            # Track position for logging
            uav_positions.append(uav.position.copy().tolist())
        
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
        
        # Disabled duplicate step logging from environment since run_simulation.py already logs steps
        # if self.step_count % 10 == 0 or done or len(collision_events) > 0 or info['success']:
        #    self.logger.log_step(self.episode_num, self.step_count, actions, next_state, rewards)
        
        # Log if target is captured
        if info['success'] and not done:
            self.logger.log_capture(self.episode_num, self.step_count)
        
        # Increment episode_num when episode is done
        if done:
            self.episode_num += 1
            
        return next_state, rewards, done, info
    
    def _get_nearby_uavs(self, uav_idx):
        """Get list of UAVs near the specified UAV
        
        Args:
            uav_idx: Index of the UAV
            
        Returns:
            list: Nearby UAV objects
        """
        nearby_uavs = []
        uav = self.uavs[uav_idx]
        comm_range = CONFIG["swarm"]["federated_learning"]["communication_range"]
        
        for i, other_uav in enumerate(self.uavs):
            if i != uav_idx:  # Don't include self
                distance = np.linalg.norm(uav.position - other_uav.position)
                if distance <= comm_range:
                    nearby_uavs.append(other_uav)
                    
        return nearby_uavs
    
    def _create_ahfsi_state(self, uav_idx):
        """Create state representation for AHFSI framework
        
        Args:
            uav_idx: Index of the UAV
            
        Returns:
            dict: State representation for AHFSI framework
        """
        # Basic environment state information
        env_state = {
            "time_step": self.step_count,
            "scenario_width": self.width,
            "scenario_height": self.height,
            "uav_count": len(self.uavs),
            "obstacle_count": len(self.obstacles),
            "target_position": self.target.position.tolist() if self.target is not None else None,
            "target_velocity": self.target.velocity.tolist() if hasattr(self.target, 'velocity') else [0, 0],
            "obstacles": [
                {
                    "position": obs.position.tolist(),
                    "radius": obs.radius
                } for obs in self.obstacles
            ],
            "uavs": [
                {
                    "id": uav.id,
                    "position": uav.position.tolist(),
                    "velocity": uav.velocity.tolist()
                } for uav in self.uavs
            ]
        }
        
        return env_state
    
    def _share_swarm_information(self):
        """Share information between UAVs in the swarm
        
        This implements federated knowledge sharing between UAVs
        """
        if not self.enable_ahfsi:
            return
            
        # Clear previous messages
        for uav in self.uavs:
            if hasattr(uav, 'received_messages'):
                uav.received_messages = []
            
        # Each UAV shares information with others in range
        total_messages = 0
        for i, uav in enumerate(self.uavs):
            # Get nearby UAVs
            nearby_uavs = self._get_nearby_uavs(i)
            
            # Share information with nearby UAVs
            if hasattr(uav, 'share_information'):
                messages_sent = uav.share_information(nearby_uavs, 
                                                   max_range=CONFIG["swarm"]["federated_learning"]["communication_range"])
                total_messages += messages_sent
        
    def _get_target_evasive_action(self):
        """Simplified evasion strategy: move away from UAVs and maintain maximum speed.
        
        Returns:
            numpy.ndarray: Force vector for target evasion
        """
        # Initialize evasion force
        direction = np.zeros(2, dtype=np.float32)
        
        # Calculate direction vector away from UAVs
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
            "success": False,
            "reward_components": [],
            "swarm_metrics": {}
        }
        
        # Add swarm intelligence metrics if enabled
        if self.enable_ahfsi:
            # Calculate coverage efficiency if information theory is available
            if self.information_theory is not None:
                uav_positions = [uav.position for uav in self.uavs]
                coverage = self.information_theory.calculate_coverage_metric(uav_positions)
                info["swarm_metrics"]["coverage"] = coverage
        
        # Initialize progress tracking if not already done
        if not hasattr(self, 'progress_metrics'):
            self.progress_metrics = {
                'min_distance': float('inf'),
                'successful_captures': 0,
                'episode_rewards': [],
                'total_steps': 0,
                'training_iterations': 0
            }
            
        # Initialize reward history for smoothing if not already done
        if not hasattr(self, 'reward_history'):
            self.reward_history = [0.0] * self.num_uavs
            
        # Initialize progressive weights if not already done
        if not hasattr(self, 'current_weights'):
            self.current_weights = {
                'approach': CONFIG["reward_weights"]["progressive"]["approach"]["initial"],
                'safety': CONFIG["reward_weights"]["progressive"]["safety"]["initial"],
                'track': CONFIG["reward_weights"]["progressive"]["track"]["initial"],
                'encircle': CONFIG["reward_weights"]["progressive"]["encircle"]["initial"],
                'capture': CONFIG["reward_weights"]["progressive"]["capture"]["initial"],
                'finish': CONFIG["reward_weights"]["progressive"]["finish"]["initial"]
            }
            
        # Update progressive weights only in training mode
        self.progress_metrics['total_steps'] += 1
        if self.mode == "train" and self.progress_metrics['total_steps'] % 100 == 0:
            # Update weights every 100 steps during training only
            for key in self.current_weights.keys():
                initial = CONFIG["reward_weights"]["progressive"][key]["initial"]
                final = CONFIG["reward_weights"]["progressive"][key]["final"]
                decay_rate = CONFIG["reward_weights"]["progressive"][key]["decay_rate"]
                
                # Calculate weight based on exponential decay from initial to final
                progress = min(1.0, self.progress_metrics['total_steps'] / 10000)  # Cap at 10000 steps
                self.current_weights[key] = initial + (final - initial) * progress
                
            # Log weight updates only in training mode
            print(f"Updated reward weights at step {self.progress_metrics['total_steps']}: {self.current_weights}")
        
        # Calculate distances from UAVs to target
        current_distances = [np.linalg.norm(uav.position - self.target.position) for uav in self.uavs]
        
        # Track minimum distance for progress reporting
        current_min_distance = min(current_distances)
        if current_min_distance < self.progress_metrics['min_distance']:
            self.progress_metrics['min_distance'] = current_min_distance
            print(f"New minimum distance to target: {current_min_distance:.4f}")
        
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
            
            # Track successful captures for progress reporting
            self.progress_metrics['successful_captures'] += 1
            print(f"Successful capture! Total captures: {self.progress_metrics['successful_captures']}")
        
        info["task_state"] = task_state
        self.task_state = task_state
        
        # Calculate rewards for each UAV based on the curriculum
        for i, uav in enumerate(self.uavs):
            # Base rewards common to all tasks - with normalization
            approach_reward = self._calculate_approach_reward(uav) * 0.5  # Scale down approach
            safety_reward = self._calculate_safety_reward(uav) * 2.0      # Scale up safety
            
            # Task-specific reward based on curriculum learning
            task_reward = 0
            task_reward_type = "track"  # Default
            
            if task_state == "tracking":
                # Tracking reward: negative sum of distances (encourage getting closer)
                normalized_distance = total_distance / (np.sqrt(self.width**2 + self.height**2) * self.num_uavs)
                task_reward = -normalized_distance  # Normalized to [-1, 0]
                task_reward_type = "track"
                
            elif task_state == "encircling":
                # Encircling reward: based on formation of encirclement
                if self.num_uavs >= 3:
                    # The paper uses ln(sum of triangles - total encirclement + 1)
                    encircle_diff = abs(sum(triangle_areas) - total_encirclement_area)
                    max_possible_diff = self.width * self.height  # Maximum possible difference
                    normalized_diff = encircle_diff / max_possible_diff
                    task_reward = -np.log(normalized_diff + 1) / self.num_uavs
                    # Clip to prevent extreme values
                    task_reward = np.clip(task_reward, -2.0, 2.0)
                task_reward_type = "encircle"
                
            elif task_state == "capturing":
                # Capturing reward: based on shrinking the encirclement
                if self._prev_distances is not None:
                    # Positive reward if getting closer to target
                    distance_change = sum(self._prev_distances) - sum(current_distances)
                    normalized_change = distance_change / (self.num_uavs * CONFIG["uav_max_velocity"])
                    task_reward = np.exp(normalized_change) - 1  # Ensure 0 reward for no change
                    # Clip to prevent extreme values
                    task_reward = np.clip(task_reward, -2.0, 2.0)
                task_reward_type = "capture"
            # Calculate basic reward components
            basic_reward = (
                self.current_weights["approach"] * approach_reward +
                self.current_weights["safety"] * safety_reward +
                self.current_weights[task_reward_type] * task_reward
            )
            
            # Add swarm behavior rewards if AHFSI is enabled
            swarm_reward = 0.0
            if self.enable_ahfsi:
                # Reward for information sharing (federated learning)
                if hasattr(uav, 'received_messages') and len(uav.received_messages) > 0:
                    # Reward for receiving information from other UAVs
                    swarm_reward += 0.05 * min(len(uav.received_messages), 5)  # Cap at 5 messages
                
                # Reward for maintaining effective formation
                if self.swarm_behavior is not None:
                    uav_positions = [u.position for u in self.uavs]
                    formation_quality = self.swarm_behavior.evaluate_formation_quality(
                        uav_positions, uav.position, task_state)
                    swarm_reward += formation_quality * 0.1
                
                # Reward for target information sharing
                if hasattr(uav, 'last_target_sighting') and uav.last_target_sighting is not None:
                    # Higher reward for recent sightings
                    time_since_sighting = self.step_count - uav.last_target_sighting["time"]
                    if time_since_sighting < 5:  # Very recent sighting
                        swarm_reward += 0.15
                    elif time_since_sighting < 20:  # Somewhat recent sighting
                        swarm_reward += 0.05
                
                # Scale swarm reward based on configuration
                swarm_factor = CONFIG["swarm"]["rl_integration"]["swarm_reward_factor"]
                swarm_reward *= swarm_factor
            
            # Add finish reward (from successful completion)
            if task_state == "finished":
                steps_remaining = CONFIG["max_steps_per_episode"] - self.step_count
                speed_bonus = steps_remaining / CONFIG["max_steps_per_episode"]  # Up to 1.0 bonus
                # Significantly increase the base finish reward to make success more valuable
                finish_reward = self.current_weights["finish"] * (5.0 + speed_bonus)
            else:
                finish_reward = 0.0
            
            # Combine all reward components
            total_reward = basic_reward + swarm_reward + finish_reward
            
            # Store the reward components for analysis and debugging
            reward_components = {
                "approach": approach_reward * self.current_weights["approach"],
                "safety": safety_reward * self.current_weights["safety"],
                task_reward_type: task_reward * self.current_weights[task_reward_type],
                "finish": finish_reward
            }
            
            # Add swarm reward to components if enabled
            if self.enable_ahfsi:
                reward_components["swarm"] = swarm_reward
                
            # Add to info for logging
            info["reward_components"].append({
                "uav_idx": i,
                "components": reward_components
            })
            
            # Apply exponential moving average for stability
            alpha = 0.8  # Weight for current reward (0.8 current, 0.2 history)
            smoothed_reward = alpha * total_reward + (1 - alpha) * self.reward_history[i]
            self.reward_history[i] = smoothed_reward
            
            rewards.append(smoothed_reward)
        
        # Track episode rewards for progress monitoring
        episode_reward_sum = sum(rewards)
        if done:
            self.progress_metrics['episode_rewards'].append(episode_reward_sum)
            avg_reward = sum(self.progress_metrics['episode_rewards'][-10:]) / min(10, len(self.progress_metrics['episode_rewards']))
            success_status = "SUCCESS" if info["success"] else "FAILED"
            print(f"Episode {self.episode_num} - {success_status} in {self.step_count} steps | Reward: {episode_reward_sum:.4f} | 10-ep avg: {avg_reward:.2f}")
        
        # Check for timeout
        if self.step_count >= CONFIG["max_steps_per_episode"]:
            done = True
            print(f"Episode timed out after {self.step_count} steps")
            # Apply a significant penalty for timeout to discourage long, unsuccessful episodes
            for i in range(len(rewards)):
                rewards[i] *= 0.5  # Halve the rewards for timeout episodes
        
        return rewards, done, info
    
    def _calculate_approach_reward(self, uav):
        """Enhanced reward for approaching the target with distance-based scaling."""
        vel_norm = np.linalg.norm(uav.velocity)
        if vel_norm < 1e-6:
            return -0.5  # Small penalty for not moving at all
        
        # Calculate distance to target (normalized)
        distance_to_target = np.linalg.norm(self.target.position - uav.position)
        max_possible_distance = np.sqrt(self.width**2 + self.height**2)
        normalized_distance = min(1.0, distance_to_target / max_possible_distance)
        
        # Calculate relative azimuth (angle between velocity and target direction)
        target_angle = math.atan2(
            self.target.position[1] - uav.position[1],
            self.target.position[0] - uav.position[0]
        )
        
        velocity_angle = math.atan2(uav.velocity[1], uav.velocity[0])
        rel_azimuth = abs(velocity_angle - target_angle) % (2 * math.pi)
        if rel_azimuth > math.pi:
            rel_azimuth = 2 * math.pi - rel_azimuth
        
        # Convert to a 0-1 range where 0 means perfect alignment and 1 means opposite direction
        normalized_alignment = rel_azimuth / math.pi
        
        # Normalize velocity (0-1 range)
        normalized_vel = vel_norm / CONFIG["uav_max_velocity"]
        
        # Calculate base approach reward
        # When perfectly aligned (normalized_alignment = 0), cos(0) = 1
        # When perpendicular (normalized_alignment = 0.5), cos(pi/2) = 0
        # When opposite (normalized_alignment = 1), cos(pi) = -1
        directional_component = math.cos(normalized_alignment * math.pi)
        
        # Apply more aggressive reward shaping using a quadratic function for better alignment
        # This more strongly rewards moving directly toward the target
        # directional_component = directional_component**3  # Cubic function emphasizes good alignment
        # If alignment is within 15 degrees, give bonus
        if normalized_alignment < 0.08:  # ~15 degrees
            directional_component += 0.5  # Bonus for very good alignment
        
        # Scale the approach reward based on distance to encourage moving at full speed when far,
        # and more careful/precise movement when close
        distance_factor = 0.5 + 0.5 * normalized_distance  # 0.5-1.0 range
        
        # Combine components
        approach_reward = normalized_vel * directional_component * distance_factor
        
        # Add progress reward if we have previous distances
        progress_bonus = 0
        if hasattr(self, '_prev_distances') and self._prev_distances is not None:
            # Get the previous distance for this UAV
            prev_distance = self._prev_distances[uav.id]
            current_distance = distance_to_target
            
            # Calculate the maximum possible progress in one step
            max_progress = CONFIG["uav_max_velocity"] * CONFIG["time_step"]
            
            # Normalize the progress to a -1 to 1 range
            normalized_progress = (prev_distance - current_distance) / max_progress
            normalized_progress = np.clip(normalized_progress, -1.0, 1.0)
            
            # Apply a bonus for making good progress toward the target
            progress_bonus = normalized_progress * 0.5  # Scale to a -0.5 to 0.5 range
        
        # Apply smoothing to avoid reward spikes
        total_approach_reward = approach_reward + progress_bonus
        if hasattr(uav, 'prev_approach_reward'):
            total_approach_reward = 0.8 * total_approach_reward + 0.2 * uav.prev_approach_reward
        
        # Store for next iteration
        uav.prev_approach_reward = total_approach_reward
        
        return total_approach_reward
    
    def _calculate_safety_reward(self, uav):
        """Enhanced safety reward with velocity-based scaling and proximity penalties."""
        # Get UAV velocity magnitude for scaling penalties
        velocity_magnitude = np.linalg.norm(uav.velocity)
        velocity_factor = 1.0 + (velocity_magnitude / CONFIG["uav_max_velocity"])  # 1.0-2.0 range
        
        # Initialize collision flags
        obstacle_collision = False
        boundary_collision = False
        
        # Check collision with obstacles
        for obstacle in self.obstacles:
            # Using actual physical obstacle radius without artificial enlargement
            safe_distance = obstacle.radius + CONFIG["uav_radius"] + 0.5  # Extra margin for safety
            dist_to_obstacle = np.linalg.norm(uav.position - obstacle.position) - obstacle.radius
            if dist_to_obstacle <= 0:
                obstacle_collision = True
                break
        
        # Check if out of bounds
        if (uav.position[0] < 0 or uav.position[0] > self.width or
            uav.position[1] < 0 or uav.position[1] > self.height):
            boundary_collision = True
        
        # Apply severe penalties for collisions, scaled by velocity
        if obstacle_collision:
            return -10.0 * velocity_factor  # Higher penalty at high speeds
        
        if boundary_collision:
            return -10.0 * velocity_factor  # Higher penalty at high speeds
        
        # Get minimum sensor reading for proximity calculation
        min_sensor = min(uav.sensor_data)
        sensor_range = CONFIG["sensor_range"]
        
        # Calculate proximity-based penalty with progressive scaling
        if min_sensor < sensor_range * 0.3:  # Very close to obstacle - danger zone
            # Strong exponential penalty that increases as distance decreases
            proximity_ratio = min_sensor / (sensor_range * 0.3)
            proximity_penalty = -5.0 * (1.0 - proximity_ratio)**2 * velocity_factor
        elif min_sensor < sensor_range * 0.7:  # Medium distance - caution zone
            # Moderate linear penalty
            proximity_ratio = (min_sensor - sensor_range * 0.3) / (sensor_range * 0.4)  # 0-1 range in caution zone
            proximity_penalty = -2.0 * (1.0 - proximity_ratio) * velocity_factor
        else:  # Safe distance
            # Small positive reward for maintaining safe distance
            proximity_ratio = (min_sensor - sensor_range * 0.7) / (sensor_range * 0.3)  # 0-1 range in safe zone
            proximity_penalty = 1.0 * proximity_ratio
        
        # Add small reward for slow movement near obstacles (promotes careful navigation)
        if min_sensor < sensor_range:
            caution_factor = 1.0 - (velocity_magnitude / CONFIG["uav_max_velocity"])
            caution_reward = caution_factor * (1.0 - (min_sensor / sensor_range))
        else:
            caution_reward = 0.0
        
        # Combine the components
        total_safety_reward = proximity_penalty + caution_reward
        
        # Apply smoothing to avoid reward spikes
        if hasattr(uav, 'prev_safety_reward'):
            total_safety_reward = 0.8 * total_safety_reward + 0.2 * uav.prev_safety_reward
        
        # Store for next iteration
        uav.prev_safety_reward = total_safety_reward
        
        return total_safety_reward
    
    def _evaluate_target_visibility(self, include_sensor_data=True):
        """Determine whether target is visible to each UAV
        
        Args:
            include_sensor_data: Whether to include sensor checks
            
        Returns:
            list: Boolean list of whether target is visible to each UAV
        """
        uav_can_see_target = []
        
        for uav in self.uavs:
            # Initial visibility is based on line of sight
            can_see = True
            
            # Check if any obstacle blocks the line of sight
            for obstacle in self.obstacles:
                if self._does_obstacle_block_visibility(uav.position, self.target.position, obstacle):
                    can_see = False
                    break
            
            # Add a range check
            max_sight_range = CONFIG["uav_sight_range"]
            if np.linalg.norm(uav.position - self.target.position) > max_sight_range:
                can_see = False
                
            # Additional check based on sensor data if requested
            if include_sensor_data:
                # If any sensor detects something close, it might interfere with visibility
                # This simulates sensor noise/interference when close to obstacles
                for sensor_value in uav.sensor_data:
                    # If any sensor detects an obstacle within a very close range
                    if sensor_value < CONFIG["sensor_range"] * 0.1:
                        # Reduce visibility based on proximity
                        visibility_probability = sensor_value / (CONFIG["sensor_range"] * 0.1)
                        # Apply random chance of visibility based on proximity
                        if np.random.random() > visibility_probability:
                            can_see = False
                            break
            
            uav_can_see_target.append(can_see)
            
            # Update UAV's belief state with target visibility information
            if self.enable_ahfsi and can_see and hasattr(uav, 'update_target_sighting'):
                uav.update_target_sighting(self.target.position, confidence=1.0, current_time=self.step_count)
            
        return uav_can_see_target
    
    def _does_obstacle_block_visibility(self, start_pos, end_pos, obstacle):
        """Check if an obstacle blocks the line of sight between two positions
        
        Args:
            start_pos: Starting position (e.g., UAV position)
            end_pos: End position (e.g., target position)
            obstacle: Obstacle object
            
        Returns:
            bool: True if visibility is blocked
        """
        # Vector from start to end
        line_vec = end_pos - start_pos
        line_length = np.linalg.norm(line_vec)
        line_unit_vec = line_vec / line_length if line_length > 0 else np.zeros(2)
        
        # Vector from start to obstacle center
        start_to_obstacle = obstacle.position - start_pos
        
        # Project start_to_obstacle onto line_vec to find closest point
        projection_length = np.dot(start_to_obstacle, line_unit_vec)
        
        # If projection is negative, closest point is start_pos
        # If projection is > line_length, closest point is end_pos
        # Otherwise, it's on the line segment
        if projection_length < 0 or projection_length > line_length:
            return False
            
        # Calculate closest point on line to obstacle center
        closest_point = start_pos + line_unit_vec * projection_length
        
        # Calculate distance from closest point to obstacle center
        distance = np.linalg.norm(closest_point - obstacle.position)
        
        # Using actual physical obstacle radius without artificial enlargement
        
        # Check if distance is less than obstacle radius
        return distance < obstacle.radius
    
    def _is_target_encircled(self):
        """Check if the target is inside the convex hull formed by UAVs."""
        if self.num_uavs < 3:
            return False
        
        # Get positions of UAVs that can see the target
        visible_uavs = []
        for i, uav in enumerate(self.uavs):
            distance = np.linalg.norm(uav.position - self.target.position)
            # Only count UAVs that are within encircle range
            if distance <= CONFIG["encircle_radius"] * 1.5:
                visible_uavs.append(uav.position)
        
        # Need at least 3 nearby UAVs to encircle
        if len(visible_uavs) < 3:
            return False
            
        # Convert to numpy array for easier calculations
        positions = np.array(visible_uavs)
        
        # Sort points by angle around target to check encirclement
        target_pos = self.target.position
        angles = np.arctan2(positions[:, 1] - target_pos[1], 
                           positions[:, 0] - target_pos[0])
        
        # Sort UAV positions by angle
        sorted_indices = np.argsort(angles)
        sorted_positions = positions[sorted_indices]
        
        # For consecutive groups of positions, check if target is inside any triangle
        in_polygon = False
        n = len(sorted_positions)
        
        if n >= 3:  # Need at least 3 UAVs to form any triangle
            # Check if target is inside the polygon formed by all UAVs
            # This is a ray-casting algorithm
            for i in range(n):
                j = (i + 1) % n
                
                if ((sorted_positions[i][1] > target_pos[1]) != (sorted_positions[j][1] > target_pos[1]) and
                    (target_pos[0] < (sorted_positions[j][0] - sorted_positions[i][0]) * 
                     (target_pos[1] - sorted_positions[i][1]) / 
                     (sorted_positions[j][1] - sorted_positions[i][1]) + sorted_positions[i][0])):
                    in_polygon = not in_polygon
            
            # If AHFSI is enabled, update UAVs' belief states if encirclement is detected
            if self.enable_ahfsi and in_polygon:
                for uav in self.uavs:
                    if hasattr(uav, 'belief_state'):
                        # Update belief state
                        if "encirclement" not in uav.belief_state:
                            uav.belief_state["encirclement"] = {}
                        uav.belief_state["encirclement"]["active"] = True
                        uav.belief_state["encirclement"]["time"] = self.step_count
        
        return in_polygon