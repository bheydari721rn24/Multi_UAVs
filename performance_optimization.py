# Performance Optimization for Multi-UAV Simulation
# Provides functions to improve simulation speed and reduce lag

import numpy as np
import time

# Performance monitoring variables
timing_data = {
    "physics": [],
    "rendering": [],
    "obstacle_avoidance": [],
    "sensor_update": [],
    "target_logic": []
}

# Global settings for performance optimization
performance_settings = {
    "skip_frames": 1,                # Render only every n-th frame
    "reduced_sensor_frequency": 2,   # Update sensors every n-th frame
    "enable_spatial_optimization": True, # Use spatial partitioning for faster proximity checks
    "use_simplified_physics": False, # Use simplified physics calculations (less accurate)
    "quality_preset": "balanced"     # Options: "performance", "balanced", "quality"
}

# Cached data to avoid redundant calculations
cached_data = {
    "obstacle_distances": {},
    "proximity_grid": None,
    "last_sensor_update": 0
}

# Performance presets configurations
performance_presets = {
    "performance": {
        "skip_frames": 2,
        "reduced_sensor_frequency": 3,
        "simplified_physics": True,
        "simplified_rendering": True,
        "reduced_history": True,
        "max_obstacle_check_distance": 5.0,
    },
    "balanced": {
        "skip_frames": 1,
        "reduced_sensor_frequency": 2,
        "simplified_physics": False,
        "simplified_rendering": False,
        "reduced_history": False,
        "max_obstacle_check_distance": 8.0,
    },
    "quality": {
        "skip_frames": 0,
        "reduced_sensor_frequency": 1,
        "simplified_physics": False,
        "simplified_rendering": False,
        "reduced_history": False,
        "max_obstacle_check_distance": 15.0,
    }
}

# Apply the selected performance preset
def apply_performance_preset(preset_name):
    """Apply a predefined performance preset.
    
    Args:
        preset_name: Name of the preset ("performance", "balanced", "quality")
    """
    if preset_name not in performance_presets:
        print(f"Warning: Unknown preset '{preset_name}'. Using 'balanced' instead.")
        preset_name = "balanced"
    
    preset = performance_presets[preset_name]
    
    # Update performance settings
    for key, value in preset.items():
        if key in performance_settings:
            performance_settings[key] = value
    
    performance_settings["quality_preset"] = preset_name
    
    print(f"Applied {preset_name} performance preset")
    return preset

# Create spatial partitioning grid for faster proximity checks
def create_spatial_grid(width, height, cell_size=2.0):
    """Create a spatial grid for faster proximity calculations.
    
    Args:
        width: Width of the environment
        height: Height of the environment
        cell_size: Size of each grid cell
        
    Returns:
        dict: Initialized spatial grid data structure
    """
    cols = max(1, int(width / cell_size))
    rows = max(1, int(height / cell_size))
    
    grid = {
        "cell_size": cell_size,
        "cols": cols,
        "rows": rows,
        "cells": {}
    }
    
    # Initialize empty cells
    for i in range(cols):
        for j in range(rows):
            grid["cells"][(i, j)] = []
            
    return grid

# Update spatial grid with agent positions
def update_spatial_grid(grid, uavs, obstacles, target):
    """Update spatial grid with current agent positions.
    
    Args:
        grid: The spatial grid data structure
        uavs: List of UAV objects
        obstacles: List of obstacle objects
        target: Target object
    """
    # Clear previous data
    for key in grid["cells"]:
        grid["cells"][key] = []
    
    # Helper to get grid cell coordinates
    def get_cell(position):
        x = max(0, min(grid["cols"]-1, int(position[0] / grid["cell_size"])))
        y = max(0, min(grid["rows"]-1, int(position[1] / grid["cell_size"])))
        return (x, y)
    
    # Add UAVs to grid
    for i, uav in enumerate(uavs):
        cell = get_cell(uav.position)
        grid["cells"][cell].append(("uav", i))
    
    # Add obstacles to grid
    for i, obstacle in enumerate(obstacles):
        cell = get_cell(obstacle.position)
        grid["cells"][cell].append(("obstacle", i))
    
    # Add target to grid
    cell = get_cell(target.position)
    grid["cells"][cell].append(("target", 0))

# Get nearby entities using spatial grid
def get_nearby_entities(grid, position, range_limit):
    """Get entities near a position using the spatial grid.
    
    Args:
        grid: The spatial grid data structure
        position: Position to check around
        range_limit: Maximum distance to consider
        
    Returns:
        dict: Dictionary of nearby entities by type and index
    """
    cell_range = max(1, int(range_limit / grid["cell_size"]) + 1)
    center_cell = (
        max(0, min(grid["cols"]-1, int(position[0] / grid["cell_size"]))),
        max(0, min(grid["rows"]-1, int(position[1] / grid["cell_size"])))
    )
    
    nearby = {
        "uav": [],
        "obstacle": [],
        "target": []
    }
    
    # Check cells in range
    for i in range(max(0, center_cell[0]-cell_range), 
                   min(grid["cols"], center_cell[0]+cell_range+1)):
        for j in range(max(0, center_cell[1]-cell_range), 
                       min(grid["rows"], center_cell[1]+cell_range+1)):
            if (i, j) in grid["cells"]:
                for entity_type, entity_idx in grid["cells"][(i, j)]:
                    nearby[entity_type].append(entity_idx)
    
    return nearby

# Optimized version of obstacle avoidance 
def optimized_obstacle_avoidance(uav, obstacles, original_force, grid=None):
    """Optimized version of obstacle avoidance calculation.
    
    Args:
        uav: UAV object
        obstacles: List of all obstacles
        original_force: Original control force
        grid: Optional spatial grid for optimization
        
    Returns:
        numpy.ndarray: Modified force vector with obstacle avoidance
    """
    # Start timing for performance measurement
    start_time = time.perf_counter()
    
    # Ensure original_force is a numpy array
    if not isinstance(original_force, np.ndarray):
        try:
            original_force = np.array(original_force, dtype=np.float32)
        except TypeError:
            # If conversion fails, return a default force
            print(f"Warning: Could not convert force of type {type(original_force)} to array. Using default.")
            original_force = np.array([0.0, 0.0], dtype=np.float32)
    
    # Ensure uav has position and velocity attributes
    if not hasattr(uav, 'position') or not hasattr(uav, 'velocity'):
        print(f"Warning: UAV missing required attributes. Skipping obstacle avoidance.")
        return original_force
    
    # Make sure uav.position and uav.velocity are numpy arrays
    if not isinstance(uav.position, np.ndarray):
        try:
            uav.position = np.array(uav.position, dtype=np.float32)
        except TypeError:
            print(f"Warning: UAV position is not convertible to array: {type(uav.position)}")
            return original_force
    
    if not isinstance(uav.velocity, np.ndarray):
        try:
            uav.velocity = np.array(uav.velocity, dtype=np.float32)
        except TypeError:
            print(f"Warning: UAV velocity is not convertible to array: {type(uav.velocity)}")
            return original_force
    
    # Initialize avoidance force
    avoidance_force = np.zeros(2, dtype=np.float32)
    
    # Make sure prev_avoidance_force exists
    if not hasattr(uav, 'prev_avoidance_force'):
        uav.prev_avoidance_force = np.zeros(2, dtype=np.float32)
    
    # Get preset settings
    current_preset = performance_presets[performance_settings["quality_preset"]]
    max_check_distance = current_preset["max_obstacle_check_distance"]
    
    # If we have a spatial grid, use it to get only nearby obstacles
    obstacle_indices = range(len(obstacles))
    if grid is not None and performance_settings["enable_spatial_optimization"]:
        nearby = get_nearby_entities(grid, uav.position, max_check_distance)
        obstacle_indices = nearby["obstacle"]
    
    # Get UAV sensor data - already normalized to [0,1]
    # Ensure uav.sensor_data exists (if not, create default)
    if not hasattr(uav, 'sensor_data'):
        sensor_angles_count = 16  # Default number of sensors
        uav.sensor_data = np.ones(sensor_angles_count) * max_check_distance
        print("Warning: UAV missing sensor_data, using default.")
    
    sensor_data = uav.sensor_data
    sensor_angles = np.linspace(0, 2*np.pi, len(sensor_data), endpoint=False)
    
    # Check each relevant obstacle
    for idx in obstacle_indices:
        obstacle = obstacles[idx]
        
        # Ensure obstacle has valid position and radius
        if not hasattr(obstacle, 'position') or not hasattr(obstacle, 'radius'):
            continue
        
        # Ensure obstacle position is a numpy array
        if not isinstance(obstacle.position, np.ndarray):
            try:
                obstacle.position = np.array(obstacle.position, dtype=np.float32)
            except TypeError:
                continue  # Skip this obstacle if position can't be converted
        
        # Quick distance check
        offset = obstacle.position - uav.position
        distance = np.linalg.norm(offset)
        
        # Skip if obstacle is too far (outside maximum check distance)
        if distance > max_check_distance:
            continue
            
        # Skip if obstacle is behind us and far enough
        vel_norm = np.linalg.norm(uav.velocity)
        if vel_norm > 0.01:  # Only if we're moving
            heading = uav.velocity / vel_norm  # Normalized velocity vector
            behind_factor = np.dot(heading, offset / (distance + 1e-6))
            if behind_factor < -0.5 and distance > 3.0:  # Skip obstacles that are behind us
                continue
        
        # Calculate avoidance force
        try:
            # Use actual physical radius without artificial enlargement
            safe_distance = obstacle.radius + uav.radius
        except AttributeError:
            # If radius attribute is missing, use a default
            safe_distance = 0.5
        
        # Calculate the distance factor - stronger force when closer
        # Use quadratic scaling for more responsive close-range avoidance
        proximity_factor = max(0, 1.0 - (distance / (safe_distance * 2.5))**2)
        
        if proximity_factor > 0:
            # Direction away from obstacle
            direction = uav.position - obstacle.position
            norm_direction = direction / (distance + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Calculate avoidance force - stronger when directly in front
            # This is a simplified calculation that's more efficient
            force_magnitude = proximity_factor * 3.0  # Base magnitude
            
            # Add to total avoidance force
            avoidance_force += norm_direction * force_magnitude
    
    # Apply smoothing with previous avoidance force (reduces jitter)
    try:
        smoothed_force = 0.7 * avoidance_force + 0.3 * uav.prev_avoidance_force
        # Store for next frame
        uav.prev_avoidance_force = smoothed_force.copy()
    except Exception as e:
        print(f"Warning: Smoothing failed: {e}. Using raw avoidance force.")
        smoothed_force = avoidance_force
        # Reset previous avoidance force
        uav.prev_avoidance_force = np.zeros(2, dtype=np.float32)
    
    # Combine with original force - give more weight to avoidance when close to obstacles
    avoidance_magnitude = np.linalg.norm(smoothed_force)
    if avoidance_magnitude > 0.1:
        # Dynamic weighting - more weight to avoidance when the force is strong
        avoidance_weight = min(0.8, avoidance_magnitude * 0.2)
        original_weight = 1.0 - avoidance_weight
        
        combined_force = original_weight * original_force + avoidance_weight * smoothed_force
    else:
        combined_force = original_force
    
    # Record timing
    end_time = time.perf_counter()
    timing_data["obstacle_avoidance"].append(end_time - start_time)
    
    return combined_force

# Optimized sensor update with reduced frequency
def optimized_sensor_update(uav, obstacles, width, height, step_count):
    """Update UAV sensors with reduced frequency for better performance.
    
    Args:
        uav: UAV object
        obstacles: List of obstacles
        width: Environment width
        height: Environment height
        step_count: Current simulation step count
    """
    # Get update frequency from settings
    update_frequency = performance_settings["reduced_sensor_frequency"]
    
    # Update sensors only every n-th frame or if it's the first few frames
    if step_count < 5 or step_count % update_frequency == 0:
        start_time = time.perf_counter()
        
        # Simplified sensor update logic
        num_sensors = len(uav.sensor_data)
        sensor_range = 5.0  # Default range
        
        # Reset sensor readings to maximum range
        uav.sensor_data = np.ones(num_sensors) * sensor_range
        
        # Calculate actual sensor readings based on obstacles
        for obstacle in obstacles:
            # Quick distance check first
            dist_to_center = np.linalg.norm(obstacle.position - uav.position)
            
            # Only process obstacles within range (plus obstacle radius)
            if dist_to_center > sensor_range + obstacle.radius:
                continue  # Obstacle too far to be detected
            
            # Get direction to obstacle
            dir_to_obstacle = obstacle.position - uav.position
            angle_to_obstacle = np.arctan2(dir_to_obstacle[1], dir_to_obstacle[0])
            
            # Convert to sensor index
            sensor_idx = int((angle_to_obstacle % (2 * np.pi)) / (2 * np.pi) * num_sensors) % num_sensors
            
            # Calculate actual distance to obstacle surface
            obstacle_radius = obstacle.radius  # Use actual physical size
            distance = max(0.1, dist_to_center - obstacle_radius)
            
            # Update the sensor if this reading is smaller than current
            if distance < uav.sensor_data[sensor_idx]:
                uav.sensor_data[sensor_idx] = distance
                
                # Also update adjacent sensors for smoother detection
                for offset in [-1, 1]:
                    adjacent_idx = (sensor_idx + offset) % num_sensors
                    # Increase distance slightly for adjacent sensors
                    adjacent_distance = distance * 1.1
                    if adjacent_distance < uav.sensor_data[adjacent_idx]:
                        uav.sensor_data[adjacent_idx] = adjacent_distance
        
        # Also check boundaries
        boundary_distance = min(
            uav.position[0],  # Left boundary
            width - uav.position[0],  # Right boundary
            uav.position[1],  # Bottom boundary
            height - uav.position[1]  # Top boundary
        )
        
        # If very close to boundary, update relevant sensors
        if boundary_distance < sensor_range:
            # Determine which boundary we're close to
            if uav.position[0] < sensor_range:  # Left
                angle = np.pi
            elif width - uav.position[0] < sensor_range:  # Right
                angle = 0
            elif uav.position[1] < sensor_range:  # Bottom
                angle = 3 * np.pi / 2
            elif height - uav.position[1] < sensor_range:  # Top
                angle = np.pi / 2
            else:
                angle = 0  # Default, shouldn't happen
            
            # Convert to sensor index
            sensor_idx = int((angle % (2 * np.pi)) / (2 * np.pi) * num_sensors) % num_sensors
            
            # Update sensor
            if boundary_distance < uav.sensor_data[sensor_idx]:
                uav.sensor_data[sensor_idx] = boundary_distance
                
                # Also update adjacent sensors
                for offset in [-1, 1]:
                    adjacent_idx = (sensor_idx + offset) % num_sensors
                    if boundary_distance < uav.sensor_data[adjacent_idx]:
                        uav.sensor_data[adjacent_idx] = boundary_distance * 1.1
        
        # Record timing
        end_time = time.perf_counter()
        timing_data["sensor_update"].append(end_time - start_time)

# Should the current frame be rendered based on settings
def should_render_frame(frame_count):
    """Determine if the current frame should be rendered.
    
    Args:
        frame_count: Current frame counter
        
    Returns:
        bool: True if frame should be rendered
    """
    skip_frames = performance_settings["skip_frames"]
    return skip_frames == 0 or frame_count % (skip_frames + 1) == 0

# Get current performance statistics
def get_performance_stats():
    """Get current performance statistics.
    
    Returns:
        dict: Performance statistics
    """
    stats = {}
    
    # Calculate average times for different components
    for key in timing_data:
        data = timing_data[key]
        if data:
            # Keep only last 100 entries to track recent performance
            if len(data) > 100:
                timing_data[key] = data[-100:]
                data = timing_data[key]
            
            stats[key] = {
                "avg_ms": sum(data) * 1000 / len(data),
                "max_ms": max(data) * 1000,
                "min_ms": min(data) * 1000,
                "samples": len(data)
            }
        else:
            stats[key] = {
                "avg_ms": 0,
                "max_ms": 0,
                "min_ms": 0,
                "samples": 0
            }
    
    # Add current settings
    stats["settings"] = performance_settings.copy()
    
    return stats

# Reset performance monitoring data
def reset_performance_data():
    """Reset all performance monitoring data."""
    for key in timing_data:
        timing_data[key] = []

# Print performance report
def print_performance_report():
    """Print a performance report to the console."""
    stats = get_performance_stats()
    
    print("\n===== PERFORMANCE REPORT =====")
    print(f"Quality preset: {performance_settings['quality_preset']}")
    print(f"Skip frames: {performance_settings['skip_frames']}")
    print(f"Sensor update frequency: Every {performance_settings['reduced_sensor_frequency']} frames")
    print("\nAverage execution times (ms):")
    
    for key, data in stats.items():
        if key != "settings":
            print(f"  {key.ljust(20)}: {data['avg_ms']:.2f} ms (min: {data['min_ms']:.2f}, max: {data['max_ms']:.2f}, samples: {data['samples']})")
    
    print("============================\n")

# Auto-tune performance settings based on system capability
def auto_tune_performance(env, baseline_fps=None):
    """Automatically tune performance settings based on system capability.
    
    Args:
        env: The simulation environment
        baseline_fps: Optional baseline FPS to target (default: auto-detect)
        
    Returns:
        dict: Selected performance settings
    """
    print("Auto-tuning performance settings...")
    
    # Test current performance with balanced settings
    apply_performance_preset("balanced")
    
    # Run a short benchmark
    fps_measurements = []
    start_time = time.time()
    
    # Run 10 simulation steps to measure performance
    for _ in range(10):
        # Get random actions
        actions = np.random.normal(0, 1, (env.num_uavs, 2))
        
        # Measure step time
        step_start = time.time()
        env.step(actions)
        step_time = time.time() - step_start
        
        # Calculate instantaneous FPS
        if step_time > 0:
            fps = 1.0 / step_time
            fps_measurements.append(fps)
    
    # Calculate average FPS
    avg_fps = sum(fps_measurements) / len(fps_measurements) if fps_measurements else 0
    
    # Determine appropriate preset based on performance
    if avg_fps < 15:
        # System is struggling, use performance preset
        selected_preset = "performance"
    elif avg_fps > 30:
        # System is powerful enough, use quality preset
        selected_preset = "quality"
    else:
        # Use balanced preset
        selected_preset = "balanced"
    
    # Apply the selected preset
    apply_performance_preset(selected_preset)
    
    print(f"Auto-tuning complete. Detected average FPS: {avg_fps:.1f}, selected preset: {selected_preset}")
    return performance_settings
