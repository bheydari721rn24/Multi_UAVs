# Agent implementations for Multi-UAV simulation system
# Contains classes for UAVs, Target, and Obstacles with physics and sensor models

import numpy as np
import math
import os
import sys
from utils import CONFIG

# Import AHFSI framework components if available
try:
    from ahfsi_framework import AHFSIController
    AHFSI_AVAILABLE = True
except ImportError as e:
    AHFSI_AVAILABLE = False
    print(f"Warning: AHFSI framework not available. Error: {e}. Running with standard behavior.")  # Import global configuration parameters

class UAV:
    """Unmanned Aerial Vehicle (UAV) agent with physics-based movement and sensor capabilities.
    
    The UAV includes realistic physics simulation (position, velocity, acceleration),
    obstacle sensors, and tracking capabilities. UAVs can detect obstacles at a distance
    and implement avoidance behaviors through sensor data.
    """
    def __init__(self, id, initial_position, color='red', enable_ahfsi=True):
        """Initialize a UAV with a unique ID and starting position.
        
        Args:
            id: Unique identifier for this UAV
            initial_position: Starting [x,y] coordinates
            color: Color for visualization (default: red)
            enable_ahfsi: Whether to enable AHFSI framework integration
        """
        self.id = id  # Unique identifier for this UAV
        self.position = np.array(initial_position, dtype=np.float32)  # Position in 2D space (km)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # Velocity vector (km/s)
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)  # Acceleration vector (km/s²)
        self.color = color  # Color for visualization
        # Initialize sensors to max range - for detecting obstacles in multiple directions
        self.sensor_data = np.zeros(CONFIG["num_sensors"]) + CONFIG["sensor_range"] 
        self.yaw = 0.0  # Orientation in radians
        
        # Store position history for trajectory visualization
        # (Note: black trajectory visualization has been disabled as requested)
        self.trajectory = [self.position.copy()]
        
        # Store previous avoidance force for smooth transitions between timesteps
        # This prevents abrupt direction changes when avoiding obstacles
        self.prev_avoidance_force = np.array([0.0, 0.0], dtype=np.float32)
        
        # Initialize AHFSI controller if available and enabled
        self.ahfsi_controller = None
        if AHFSI_AVAILABLE and enable_ahfsi and CONFIG["swarm"]["rl_integration"]["enabled"]:
            self.ahfsi_controller = AHFSIController(id)
            
        # Belief state maintained by the UAV
        self.belief_state = {"target": None, "obstacles": [], "confidence": 0.5}
        
        # Communication capabilities
        self.received_messages = []
        self.last_target_sighting = None
        
    def update(self, force, dt=CONFIG["time_step"]):
        """Update UAV physics (position, velocity, acceleration) based on applied force.
        
        Implements realistic Newtonian physics with appropriate limits on acceleration
        and velocity to model UAV movement capabilities. Position history is recorded
        for visualization purposes.
        
        Args:
            force: Applied force vector [x,y] in Newtons
            dt: Time step in seconds
        """
        # Calculate acceleration using Newton's Second Law (F = ma, so a = F/m)
        self.acceleration = force / CONFIG["uav_mass"]
        
        # Limit acceleration to maximum UAV capabilities
        acc_norm = np.linalg.norm(self.acceleration)
        if acc_norm > CONFIG["uav_max_acceleration"]:
            self.acceleration = self.acceleration / acc_norm * CONFIG["uav_max_acceleration"]
        
        # Update velocity using physics equation: v = v₀ + a·t
        self.velocity += self.acceleration * dt
        
        # Limit velocity to maximum UAV capabilities
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > CONFIG["uav_max_velocity"]:
            self.velocity = self.velocity / vel_norm * CONFIG["uav_max_velocity"]
        
        # Update position using physics equation: p = p₀ + v·t
        self.position += self.velocity * dt
        
        # Calculate yaw/heading angle of UAV based on velocity direction
        # This aligns the UAV graphic with its movement direction
        if vel_norm > 0:
            self.yaw = math.atan2(self.velocity[1], self.velocity[0])
        
        # Update position history for trajectory visualization
        # Though trajectory drawing is disabled, we still maintain the data
        self.trajectory.append(self.position.copy())
        # Maintain fixed-length trajectory history to avoid memory issues
        if len(self.trajectory) > CONFIG["trajectory_length"]:
            self.trajectory.pop(0)
    
    def get_state(self, scenario_width, scenario_height):
        # Normalized position and velocity as in equation (11)
        state = np.array([
            self.position[0] / scenario_width,
            self.position[1] / scenario_height,
            self.velocity[0] / CONFIG["uav_max_velocity"],
            self.velocity[1] / CONFIG["uav_max_velocity"]
        ], dtype=np.float32)
        
        # If AHFSI is enabled, augment state with swarm intelligence information
        if self.ahfsi_controller is not None:
            state = self.ahfsi_controller.get_augmented_state(state)
            
        return state
    
    def get_normalized_sensor_data(self):
        return np.clip(self.sensor_data / CONFIG["sensor_range"], 0, 1)
        
    def get_observation(self):
        """Get the observation vector for reinforcement learning.
        
        Returns:
            numpy.ndarray: Observation vector containing position, velocity, and sensor data
        """
        # Include normalized position in scenario
        scenario_width = CONFIG["scenario_width"]
        scenario_height = CONFIG["scenario_height"]
        pos_x_norm = self.position[0] / scenario_width
        pos_y_norm = self.position[1] / scenario_height
        
        # Include normalized velocity
        vel_x_norm = self.velocity[0] / CONFIG["uav_max_velocity"]
        vel_y_norm = self.velocity[1] / CONFIG["uav_max_velocity"]
        
        # Include normalized sensor data for obstacle detection
        sensor_data = self.get_normalized_sensor_data()
        
        # Combine all observations
        obs = np.concatenate(
            [
                [pos_x_norm, pos_y_norm],  # Position (2)
                [vel_x_norm, vel_y_norm],  # Velocity (2)
                sensor_data,  # Sensor readings (CONFIG["num_sensors"])
            ]
        )
        
        # Add AHFSI belief information if available
        if hasattr(self, 'belief_state') and self.belief_state["target"] is not None:
            # Target belief normalized position
            target_belief = np.array(self.belief_state["target"]) / np.array([scenario_width, scenario_height])
            target_confidence = np.array([self.belief_state["confidence"]])
            obs = np.concatenate([obs, target_belief, target_confidence])
        
        return obs      
    def process_rl_action(self, action, nearby_uavs=None, env_state=None):
        """Process RL action through AHFSI framework if enabled
        
        Args:
            action: Action from RL policy
            nearby_uavs: List of nearby UAVs (optional)
            env_state: Environment state (optional)
            
        Returns:
            numpy.ndarray: Processed force vector
        """
        if self.ahfsi_controller is not None and nearby_uavs is not None and env_state is not None:
            # Process action through AHFSI framework
            return self.ahfsi_controller.process_rl_action(self, nearby_uavs, action, env_state)
        else:
            # Standard behavior - convert action directly to force
            return action * CONFIG["uav_max_acceleration"]
            
    def receive_message(self, message):
        """Receive message from another UAV
        
        Args:
            message: Message data
        """
        # Store message
        self.received_messages.append(message)
        
        # Process message if it contains target information
        if "target_detection" in message and message["target_detection"] is not None:
            target_info = message["target_detection"]
            
            # Update last target sighting if newer or more confident
            if self.last_target_sighting is None or \
               target_info["time"] > self.last_target_sighting["time"] or \
               (target_info["time"] == self.last_target_sighting["time"] and \
                target_info["confidence"] > self.last_target_sighting["confidence"]):
                
                self.last_target_sighting = target_info
                
        # Update belief state with message sender position
        if "sender_id" in message and "position" in message:
            sender_id = message["sender_id"]
            self.belief_state[f"uav_{sender_id}_position"] = message["position"]
            
    def update_target_sighting(self, target_position, confidence=1.0, current_time=0):
        """Update target sighting information
        
        Args:
            target_position: Target position
            confidence: Confidence in the sighting (0-1)
            current_time: Current simulation time
        """
        self.last_target_sighting = {
            "position": target_position,
            "time": current_time,
            "confidence": confidence
        }
        
        # Update belief state
        self.belief_state["target"] = {
            "position": target_position,
            "confidence": confidence,
            "last_update": current_time
        }
        
    def share_information(self, nearby_uavs, max_range=3.0):
        """Share information with nearby UAVs
        
        Args:
            nearby_uavs: List of nearby UAVs
            max_range: Maximum communication range
            
        Returns:
            int: Number of messages sent
        """
        message_count = 0
        
        # Create basic message
        message = {
            "sender_id": self.id,
            "position": self.position,
            "velocity": self.velocity,
            "sensor_data": self.sensor_data.tolist(),
            "target_detection": self.last_target_sighting
        }
        
        # Share with nearby UAVs within communication range
        for uav in nearby_uavs:
            if uav.id != self.id:  # Don't share with self
                distance = np.linalg.norm(self.position - uav.position)
                if distance <= max_range:
                    # Send message
                    if hasattr(uav, 'receive_message'):
                        uav.receive_message(message)
                        message_count += 1
                        
        return message_count
    
    def update_sensors(self, obstacles, scenario_width, scenario_height):
        # Reset sensors to max range
        self.sensor_data = np.zeros(CONFIG["num_sensors"]) + CONFIG["sensor_range"]
        
        # Update sensor readings based on obstacles and boundaries
        for i in range(CONFIG["num_sensors"]):
            angle = 2 * math.pi * i / CONFIG["num_sensors"]
            sensor_dir = np.array([math.cos(angle), math.sin(angle)])
            
            # Check for obstacles
            for obstacle in obstacles:
                obs_pos = obstacle.position
                # Use the actual physical obstacle radius without artificial enlargement
                obs_radius = obstacle.radius
                obs_vec = obs_pos - self.position
                
                # Project obstacle vector onto sensor direction
                proj_length = np.dot(obs_vec, sensor_dir)
                if proj_length > 0:  # Only consider obstacles in front of the sensor
                    # Find closest point on sensor ray to obstacle center
                    closest_point = self.position + sensor_dir * proj_length
                    distance_to_center = np.linalg.norm(closest_point - obs_pos)
                    
                    if distance_to_center < obs_radius:
                        # Ray intersects with obstacle
                        # Calculate the exact intersection point
                        delta = math.sqrt(max(0, obs_radius**2 - distance_to_center**2))
                        intersection_dist = proj_length - delta
                        if 0 < intersection_dist < self.sensor_data[i]:
                            self.sensor_data[i] = intersection_dist
            
            # Check for boundaries
            # Left boundary
            if sensor_dir[0] < 0:
                dist = self.position[0] / abs(sensor_dir[0])
                if dist < self.sensor_data[i]:
                    self.sensor_data[i] = dist
            # Right boundary
            elif sensor_dir[0] > 0:
                dist = (scenario_width - self.position[0]) / sensor_dir[0]
                if dist < self.sensor_data[i]:
                    self.sensor_data[i] = dist
            # Bottom boundary
            if sensor_dir[1] < 0:
                dist = self.position[1] / abs(sensor_dir[1])
                if dist < self.sensor_data[i]:
                    self.sensor_data[i] = dist
            # Top boundary
            elif sensor_dir[1] > 0:
                dist = (scenario_height - self.position[1]) / sensor_dir[1]
                if dist < self.sensor_data[i]:
                    self.sensor_data[i] = dist

    def update(self, force, dt=CONFIG["time_step"]):
        # F = ma, so a = F/m
        self.acceleration = force / CONFIG["uav_mass"]
        
        # Clamp acceleration
        acc_norm = np.linalg.norm(self.acceleration)
        if acc_norm > CONFIG["uav_max_acceleration"]:
            self.acceleration = self.acceleration / acc_norm * CONFIG["uav_max_acceleration"]
        
        # Update velocity: v = v0 + a*t
        self.velocity += self.acceleration * dt
        
        # Clamp velocity
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > CONFIG["uav_max_velocity"]:
            self.velocity = self.velocity / vel_norm * CONFIG["uav_max_velocity"]
        
        # Update position: p = p0 + v*t
        self.position += self.velocity * dt
        
        # Calculate yaw angle (in radians)
        if vel_norm > 0:
            self.yaw = math.atan2(self.velocity[1], self.velocity[0])
        
        # Update trajectory for visualization
        self.trajectory.append(self.position.copy())
        # Keep trajectory length limited
        if len(self.trajectory) > CONFIG["trajectory_length"]:
            self.trajectory.pop(0)
            
        # Update belief state with new position
        self.belief_state["position"] = self.position.copy()

class Target:
    def __init__(self, initial_position):
        self.position = np.array(initial_position, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)
        self.yaw = 0.0
        
        # For trajectory visualization
        self.trajectory = [self.position.copy()]
        
        # Boundary avoidance parameters - made much stronger
        self.boundary_margin = CONFIG["scenario_width"] * 0.05  # Keep at least 25% away from borders
        self.boundary_force_factor = 8.0  # Very strong repulsion force
    
    def avoid_boundaries(self, scenario_width, scenario_height):
        """Simple, direct approach to avoid boundaries - much more reliable"""
        # Calculate distance to each boundary
        dist_left = self.position[0]  # Distance to left boundary
        dist_right = scenario_width - self.position[0]  # Distance to right boundary
        dist_bottom = self.position[1]  # Distance to bottom boundary
        dist_top = scenario_height - self.position[1]  # Distance to top boundary
        
        # Initialize avoidance force
        avoidance_force = np.zeros(2, dtype=np.float32)
        
        # Check if near left boundary
        if dist_left < self.boundary_margin:
            # Force to the right, stronger when closer
            repulsion = (1.0 - dist_left/self.boundary_margin) * self.boundary_force_factor
            avoidance_force[0] += repulsion * CONFIG["target_max_acceleration"]
        
        # Check if near right boundary
        if dist_right < self.boundary_margin:
            # Force to the left, stronger when closer
            repulsion = (1.0 - dist_right/self.boundary_margin) * self.boundary_force_factor
            avoidance_force[0] -= repulsion * CONFIG["target_max_acceleration"]
        
        # Check if near bottom boundary
        if dist_bottom < self.boundary_margin:
            # Force upward, stronger when closer
            repulsion = (1.0 - dist_bottom/self.boundary_margin) * self.boundary_force_factor
            avoidance_force[1] += repulsion * CONFIG["target_max_acceleration"]
        
        # Check if near top boundary
        if dist_top < self.boundary_margin:
            # Force downward, stronger when closer
            repulsion = (1.0 - dist_top/self.boundary_margin) * self.boundary_force_factor
            avoidance_force[1] -= repulsion * CONFIG["target_max_acceleration"]
        
        # Apply stronger force when very close to any boundary
        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
        danger_zone = self.boundary_margin * 0.5
        if min_dist < danger_zone:
            # Double the force when in danger zone
            emergency_factor = 2.0 * (1.0 - min_dist/danger_zone) 
            avoidance_force *= (1.0 + emergency_factor)
        
        # Limit the maximum force if needed
        force_magnitude = np.linalg.norm(avoidance_force)
        max_force = CONFIG["target_max_acceleration"] * 2.0
        if force_magnitude > max_force:
            avoidance_force = avoidance_force / force_magnitude * max_force
        
        return avoidance_force
        
    def update(self, force, dt=CONFIG["time_step"], scenario_width=CONFIG["scenario_width"], scenario_height=CONFIG["scenario_height"]):
        # Get boundary avoidance force using the new simpler method
        boundary_force = self.avoid_boundaries(scenario_width, scenario_height)
        
        # Combine original force with boundary avoidance - give more weight to boundary avoidance
        combined_force = force * 0.3 + boundary_force * 0.7
        
        # Update acceleration with combined force
        self.acceleration = combined_force
        
        # Clamp acceleration
        acc_norm = np.linalg.norm(self.acceleration)
        if acc_norm > CONFIG["target_max_acceleration"]:
            self.acceleration = self.acceleration / acc_norm * CONFIG["target_max_acceleration"]
        
        # Update velocity: v = v0 + a*t
        self.velocity += self.acceleration * dt
        
        # Clamp velocity
        vel_norm = np.linalg.norm(self.velocity)
        if vel_norm > CONFIG["target_max_velocity"]:
            self.velocity = self.velocity / vel_norm * CONFIG["target_max_velocity"]
        
        # Update position: p = p0 + v*t
        self.position += self.velocity * dt
        
        # Final safety check - ensure position stays within boundaries with a buffer
        buffer = CONFIG["target_radius"] * 2
        
        # Left boundary
        if self.position[0] < buffer:
            self.position[0] = buffer
            self.velocity[0] = abs(self.velocity[0]) * 0.5  # Reflect and strongly dampen
            
        # Right boundary
        elif self.position[0] > scenario_width - buffer:
            self.position[0] = scenario_width - buffer
            self.velocity[0] = -abs(self.velocity[0]) * 0.5  # Reflect and strongly dampen
            
        # Bottom boundary
        if self.position[1] < buffer:
            self.position[1] = buffer
            self.velocity[1] = abs(self.velocity[1]) * 0.5  # Reflect and strongly dampen
            
        # Top boundary
        elif self.position[1] > scenario_height - buffer:
            self.position[1] = scenario_height - buffer
            self.velocity[1] = -abs(self.velocity[1]) * 0.5  # Reflect and strongly dampen
        
        # Calculate yaw angle (in radians)
        if vel_norm > 0:
            self.yaw = math.atan2(self.velocity[1], self.velocity[0])
        
        # Update trajectory for visualization
        self.trajectory.append(self.position.copy())
        # Keep trajectory length limited
        if len(self.trajectory) > CONFIG["trajectory_length"]:
            self.trajectory.pop(0)
    
    def get_state(self, scenario_width, scenario_height):
        return np.array([
            self.position[0] / scenario_width,
            self.position[1] / scenario_height,
            self.velocity[0] / CONFIG["target_max_velocity"],
            self.velocity[1] / CONFIG["target_max_velocity"]
        ], dtype=np.float32)

class Obstacle:
    def __init__(self, position, radius):
        self.position = np.array(position, dtype=np.float32)
        self.radius = radius
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)  # For dynamic obstacles