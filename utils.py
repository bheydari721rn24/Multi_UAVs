import numpy as np

# Configuration parameters
CONFIG = {
    # Environment parameters
    "scenario_width": 10.0,  # 2km
    "scenario_height": 10.0,
    "uav_radius": 0.05,
    "target_radius": 0.05,
    "obstacle_radius_min": 0.04,
    "obstacle_radius_max": 0.1,
    "capture_distance": 1.2,  # Effective round-up range of a single UAV
    "sensor_range": 0.4,  # Detection range of sensors
    "max_steps_per_episode": 50,
    "num_sensors": 24,  # Number of range sensors per UAV
    "trajectory_length": 50,  # Length of trajectory to display
    "time_step": 1,  # Simulation time step
    
    # UAV parameters
    "uav_initial_velocity": 0.0,
    "uav_max_velocity": 0.13,  # 0.1 km/s
    "uav_max_acceleration": 0.05,  # 0.04 km/s²
    "uav_mass": 0.5,  # Mass for force calculation
    
    # Target parameters
    "target_initial_velocity": 0.0,
    "target_max_velocity": 0.13,  # 0.13 km/s (faster than UAVs)
    "target_max_acceleration": 0.05,  # 0.05 km/s²
    
    # Training parameters
    "replay_buffer_size": 100000,
    "batch_size": 256,
    "pre_batch_size": 50,  # Size of first sampling
    "gamma": 0.99,  # Discount factor
    "tau": 0.001,  # Soft update parameter
    "actor_lr": 0.0005,
    "critic_lr": 0.001,
    "num_episodes": 30000,
    "epsilon": 1.0,  # Starting value for epsilon
    "epsilon_min": 0.01,  # Ending value for epsilon
    "epsilon_decay": 0.98,  # Decay rate for epsilon
    
    # Curriculum learning parameters
    "d_limit": 1.5,  # 10 * capture_distance
    "curriculum_threshold": 5000,  # Number of training steps before moving to next curriculum level
    "curriculum_step_size": 1000,  # Step size for curriculum levels
    "curriculum_learning": {
        "reward_weights": {
            "approach": 1.0,
            "safety": 0.5,
            "track": 0.3,
            "encircle": 0.5,
            "capture": 1.0,
            "finish": 10.0
        },
        "curriculum_steps": [
            {"step": 0, "reward_weights": {"approach": 1.0, "safety": 0.5, "track": 0.3, "encircle": 0.5, "capture": 1.0, "finish": 10.0}},
            {"step": 1000, "reward_weights": {"approach": 0.8, "safety": 0.6, "track": 0.4, "encircle": 0.6, "capture": 1.2, "finish": 12.0}},
            {"step": 2000, "reward_weights": {"approach": 0.6, "safety": 0.7, "track": 0.5, "encircle": 0.7, "capture": 1.4, "finish": 14.0}},
            {"step": 3000, "reward_weights": {"approach": 0.4, "safety": 0.8, "track": 0.6, "encircle": 0.8, "capture": 1.6, "finish": 16.0}},
            {"step": 4000, "reward_weights": {"approach": 0.2, "safety": 0.9, "track": 0.7, "encircle": 0.9, "capture": 1.8, "finish": 18.0}},
        ],
    },
    
    # Correlation weights for the correlation index function
    "correlation_weights": {
        "sigma1": 100,
        "sigma2": 2,
        "sigma3": 5
    },
    
    # Reward weights
    "reward_weights": {
        "approach": 1.0,
        "safety": 1.0,
        "track": 1.0,
        "encircle": 1.0,
        "capture": 1.0,
        "finish": 1.0
    }
}

def calculate_triangle_area(p1, p2, p3):
    """Calculate the area of a triangle using cross product."""
    v1 = p1 - p3
    v2 = p2 - p3
    cross = np.cross(v1, v2)
    return 0.5 * abs(cross)

def point_in_triangle(p, p1, p2, p3):
    """Check if point p is inside triangle p1-p2-p3 using barycentric coordinates."""
    def area(x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
    
    a = area(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
    a1 = area(p[0], p[1], p2[0], p2[1], p3[0], p3[1])
    a2 = area(p1[0], p1[1], p[0], p[1], p3[0], p3[1])
    a3 = area(p1[0], p1[1], p2[0], p2[1], p[0], p[1])
    
    # Check if sum of sub-triangle areas equals the total triangle area
    return abs(a - (a1 + a2 + a3)) < 1e-9

def create_directory(path):
    """Create directory if it doesn't exist."""
    import os
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")