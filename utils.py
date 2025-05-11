import numpy as np


# Configuration parameters
CONFIG = {
    # Environment parameters
    "scenario_width": 15.0,  # 15km - width of simulation environment
    "scenario_height": 15.0,  # 15km - height of simulation environment
    "uav_radius": 0.05,  # 0.05km (50m) - radius of UAV physical boundary
    "target_radius": 0.05,  # 0.05km (50m) - radius of target physical boundary
    "obstacle_radius_min": 0.4,  # 0.4km (400m) - minimum obstacle radius
    "obstacle_radius_max": 3.0,  # 3.0km (3000m) - maximum obstacle radius
    "capture_distance": 1.5,  # Increased from 1.2 to make capture easier and improve success rate
    "encircle_radius": 2.5,  # Radius for UAV encirclement formation around target (km)
    "sensor_range": 0.5,  # Increased from 0.4 to improve obstacle detection with enhanced avoidance
    "max_steps_per_episode": 50,  # Increased from 50 to allow for more exploration and learning
    "num_sensors": 24,  # Number of range sensors per UAV for comprehensive 360-degree detection
    "trajectory_length": 50,  # Length of trajectory data stored (matching max_steps_per_episode)
    "time_step": 1,  # 1 second per simulation step
    
    # UAV parameters
    "uav_initial_velocity": 0.0,  # UAVs start stationary
    "uav_max_velocity": 0.13,  # 0.13 km/s (130 m/s, ~470 km/h) maximum velocity
    "uav_max_acceleration": 0.05,  # 0.05 km/s² (50 m/s²) maximum acceleration
    "uav_mass": 0.5,  # 0.5 (arbitrary unit) - mass used for force calculations
    "uav_sight_range": 3.0,  # 3.0 km - maximum distance for target visibility
    
    # Target parameters
    "target_initial_velocity": 0.0,  # Target starts stationary
    "target_max_velocity": 0.13,  # 0.13 km/s (130 m/s, ~470 km/h) - matches UAV maximum
    "target_max_acceleration": 0.05,  # 0.05 km/s² (50 m/s²) - matches UAV maximum
    
    # Training parameters
    "replay_buffer_size": 100000,  # Memory buffer size for experience replay
    "batch_size": 256,  # Number of samples per training batch
    "pre_batch_size": 64,  # Increased from 50 for more stable initial training
    "gamma": 0.99,  # Discount factor for future rewards
    "tau": 0.001,  # Soft update parameter for target networks
    "actor_lr": 0.0003,  # Actor learning rate - reduced from 0.0005 for stability
    "critic_lr": 0.0007,  # Critic learning rate - reduced from 0.001 for stability
    "train_interval": 5,  # Train the network every 5 simulation steps
    "num_episodes": 2,  # Increased from 2 to allow for proper model differentiation
    "epsilon": 1.0,  # Starting exploration rate (100% random actions)
    "epsilon_min": 0.05,  # Minimum exploration rate (5% random actions)
    "epsilon_decay": 0.995,  # Exploration decay factor - slower for better learning
    
    # Curriculum learning parameters
    "d_limit": 2.0,  # Distance threshold (km) - increased from 1.5 for easier tracking
    "curriculum_threshold": 3000,  # Performance threshold for curriculum advancement
    "curriculum_step_size": 500,  # Step size for curriculum difficulty increases
    "curriculum_learning": {
        "reward_weights": {
            "approach": 1.0,  # Base weight for approaching target
            "safety": 0.5,   # Base weight for maintaining safety from obstacles
            "track": 0.5,    # Base weight for tracking - increased from 0.3 for better pursuit
            "encircle": 0.7, # Base weight for encirclement - increased from 0.5 to promote cooperation
            "capture": 1.5,  # Base weight for capture - increased from 1.0 to prioritize this behavior
            "finish": 15.0   # Base weight for task completion - increased from 10.0 for stronger success signal
        },
        # Progression of reward weights throughout curriculum learning
        # Gradually decreases approach weight while increasing safety, tracking, encirclement, capture and finish
        "curriculum_steps": [
            # Initial stage - focus on approach and basic task completion
            {"step": 0, "reward_weights": {"approach": 1.0, "safety": 0.5, "track": 0.5, "encircle": 0.7, "capture": 1.5, "finish": 15.0}},
            # Stage 1 - slightly more emphasis on safety and complex behaviors
            {"step": 500, "reward_weights": {"approach": 0.9, "safety": 0.6, "track": 0.6, "encircle": 0.8, "capture": 1.8, "finish": 17.0}},
            # Stage 2 - balanced approach and more complex behaviors
            {"step": 1000, "reward_weights": {"approach": 0.8, "safety": 0.7, "track": 0.7, "encircle": 0.9, "capture": 2.0, "finish": 19.0}},
            # Stage 3 - prioritizing safety and coordination behaviors
            {"step": 1500, "reward_weights": {"approach": 0.7, "safety": 0.8, "track": 0.8, "encircle": 1.0, "capture": 2.2, "finish": 21.0}},
            # Stage 4 - higher emphasis on complex behaviors
            {"step": 2000, "reward_weights": {"approach": 0.6, "safety": 0.9, "track": 0.9, "encircle": 1.1, "capture": 2.4, "finish": 23.0}},
            # Final stage - safety and complex team behaviors maximized
            {"step": 2500, "reward_weights": {"approach": 0.5, "safety": 1.0, "track": 1.0, "encircle": 1.2, "capture": 2.6, "finish": 25.0}},
        ],
    },
    
    # Correlation weights for the correlation index function - controls multi-agent coordination
    "correlation_weights": {
        "sigma1": 100,  # Weight for encirclement formation quality (higher value = more important)
        "sigma2": 2,    # Weight for overall encirclement size (lower value = less emphasis)
        "sigma3": 5     # Weight for UAV spread/distribution (balanced value for formation)
    },
    
    # Reward weights - Enhanced with progressive structure for dynamic learning
    "reward_weights": {
        # Base weights (backwards compatible with older implementations)
        "approach": 1.0,  # Reward for moving toward target
        "safety": 1.0,   # Reward for avoiding obstacles (with enhanced 10x obstacle collision prevention)
        "track": 1.0,    # Reward for maintaining good relative position to target
        "encircle": 1.0, # Reward for coordinated surrounding of target
        "capture": 1.0,  # Reward for being close enough to capture
        "finish": 10.0,  # Reward for successfully completing mission
        
        # Progressive weights structure - includes initial value, final value, and decay rate
        # These values adapt during training to guide learning from simpler to more complex behaviors
        "progressive": {
            "approach": {"initial": 1.0, "final": 0.5, "decay_rate": 0.9998},  # Decreases over time as other behaviors become more important
            "safety": {"initial": 0.6, "final": 0.9, "decay_rate": 0.9998},    # Increases to emphasize proactive obstacle avoidance with sensor data
            "track": {"initial": 0.5, "final": 1.0, "decay_rate": 0.9998},     # Increases to improve following behavior
            "encircle": {"initial": 0.7, "final": 1.5, "decay_rate": 0.9998},  # Increases to encourage better coordination between UAVs
            "capture": {"initial": 1.5, "final": 2.5, "decay_rate": 0.9998},   # Substantially increases to make capture the primary goal
            "finish": {"initial": 15.0, "final": 30.0, "decay_rate": 0.9998}   # Strongly reinforces successful mission completion
        }
    },
    
    # Advanced Swarm Intelligence Parameters (AHFSI Framework)
    "swarm": {
        # Federated Learning Parameters
        "federated_learning": {
            "enabled": True,
            "communication_range": 3.0,         # Maximum distance (km) for UAV communication
            "sharing_interval": 5,            # Share knowledge every N steps
            "knowledge_dimension": 128,        # Size of knowledge tensor for each UAV
            "model_fusion_weight": 0.3,       # Weight for fusing received model parameters
            "fusion_confidence_threshold": 0.6,# Minimum confidence for parameter fusion
            "federation_topology": "dynamic"   # How UAVs connect: "ring", "star", "mesh", "dynamic"
        },
        
        # Quantum-Inspired Optimization
        "quantum_optimization": {
            "enabled": True,
            "interference_factor": 0.4,        # Strength of quantum interference effects
            "phase_shift_rate": 0.15,         # Rate of phase shift in quantum calculations
            "superposition_decay": 0.97,      # Decay rate for quantum superposition
            "entanglement_strength": 0.5,     # Strength of entanglement between nearby UAVs
            "quantum_sigma": 2.5             # Parameter for quantum amplitude calculations
        },
        
        # Information-Theoretic Decision Framework
        "information_theory": {
            "enabled": True,
            "entropy_grid_size": [30, 30],     # Size of entropy grid for environment
            "entropy_decay_rate": 0.95,       # Rate of entropy decay per step
            "information_gain_weight": 0.7,    # Weight for information gain in decisions
            "exploration_factor": 0.4,        # Factor for exploration vs exploitation
            "mutual_information_threshold": 0.3 # Threshold for significant mutual information
        },
        
        # Multi-Scale Temporal Abstraction
        "temporal_abstraction": {
            "enabled": True,
            "num_temporal_scales": 3,          # Number of temporal scales for decision making
            "temporal_horizons": [5, 20, 50],  # Time horizons for different scales
            "scale_update_rates": [1.0, 0.5, 0.2], # Update rates for different temporal scales
            "temporal_discount_factors": [0.9, 0.95, 0.99] # Discount factors for each scale
        },
        
        # Game-Theoretic Multi-Agent Coordination
        "game_theory": {
            "enabled": True,
            "leader_selection_interval": 10,   # Steps between leader selection
            "leadership_quality_threshold": 0.7,# Minimum quality for leadership
            "follower_coherence_factor": 0.6,  # How strictly followers follow the leader
            "role_switching_cost": 0.3,       # Cost associated with switching roles
            "stackelberg_alpha": 0.4          # Parameter for Stackelberg equilibrium
        },
        
        # Bayesian Belief Propagation
        "belief_propagation": {
            "enabled": True,
            "belief_grid_resolution": [50, 50], # Resolution of belief grid
            "observation_noise_sigma": 0.2,    # Standard deviation of observation noise
            "process_noise_sigma": 0.1,       # Standard deviation of process noise
            "confidence_decay_rate": 0.98,     # Rate of belief confidence decay
            "min_belief_update_threshold": 0.05 # Minimum change for belief update
        },
        
        # Topological Data Analysis
        "topology": {
            "enabled": True,
            "persistence_threshold": 0.3,      # Threshold for topological feature persistence
            "filtration_steps": 10,           # Number of steps in Vietoris-Rips filtration
            "homology_dimension_max": 2,      # Maximum homology dimension to calculate
            "feature_significance_threshold": 0.4, # Threshold for significant features
            "pattern_recognition_update_rate": 0.2 # Rate for pattern recognition updates
        },
        
        # Core Swarm Behaviors
        "behaviors": {
            "cohesion": {
                "weight": 0.5,                # Base weight for cohesion
                "distance_threshold": 2.0,     # Maximum distance for cohesion effect
                "adaptive_factor": 0.2         # How much cohesion adapts based on context
            },
            "separation": {
                "weight": 0.7,                # Base weight for separation
                "minimum_distance": 0.3,       # Minimum desired separation distance
                "adaptive_factor": 0.3         # How much separation adapts based on context
            },
            "alignment": {
                "weight": 0.4,                # Base weight for velocity alignment
                "max_influence_distance": 1.5, # Maximum distance for alignment influence
                "adaptive_factor": 0.15        # How much alignment adapts based on context
            },
            "formation": {
                "weight": 0.6,                # Base weight for formation behaviors
                "formation_types": ["circle", "line", "triangle", "wedge"], # Available formations
                "formation_scale_factor": 1.2,  # Scale factor for formations
                "dynamic_formation_switching": True # Whether to switch formations dynamically
            }
        },
        
        # Integration with MADDPG
        "rl_integration": {
            "enabled": True,
            "initial_swarm_weight": 0.4,      # Initial weight for swarm behaviors vs RL
            "adaptive_weighting": True,       # Whether to adapt weights dynamically
            "context_adaptation_rate": 0.05,   # Rate of adaptation based on context
            "swarm_reward_factor": 0.3        # How much swarm behaviors affect rewards
        }
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