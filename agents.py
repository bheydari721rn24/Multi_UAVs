import numpy as np
import math
from utils import CONFIG

class UAV:
    def __init__(self, id, initial_position, color='red'):
        self.id = id
        self.position = np.array(initial_position, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.acceleration = np.array([0.0, 0.0], dtype=np.float32)
        self.color = color
        self.sensor_data = np.zeros(CONFIG["num_sensors"]) + CONFIG["sensor_range"]  # Initialize sensors to max range
        self.yaw = 0.0  # Orientation in radians
        
        # For trajectory visualization
        self.trajectory = [self.position.copy()]
        
        # For smooth motion and avoidance
        self.prev_avoidance_force = np.array([0.0, 0.0], dtype=np.float32)  # For motion smoothing
        
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
    
    def get_state(self, scenario_width, scenario_height):
        # Normalized position and velocity as in equation (11)
        return np.array([
            self.position[0] / scenario_width,
            self.position[1] / scenario_height,
            self.velocity[0] / CONFIG["uav_max_velocity"],
            self.velocity[1] / CONFIG["uav_max_velocity"]
        ], dtype=np.float32)
    
    def get_normalized_sensor_data(self):
        return self.sensor_data / CONFIG["sensor_range"]
    
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
                # Use the enlarged obstacle radius (10x) to match visualization
                obs_radius = obstacle.radius * 10
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