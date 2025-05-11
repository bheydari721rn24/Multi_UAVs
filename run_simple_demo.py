# Simple Interactive Military-Style Radar Demo
# Demonstrates UAV simulation with streamlined radar visualization

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patheffects as path_effects
import matplotlib.gridspec as gridspec
from colorama import Fore, Style, init

# Initialize colorama for terminal coloring
init()

# Add the directory containing the visualization module to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import required modules
from environment import Environment
from networks import ActorNetwork
from utils import CONFIG


# Performance tracking variables
profile_data = {
    'action_times': [],
    'env_times': [],
    'vis_times': [],
    'total_frames': 0,
    'skipped_frames': 0
}

# Simple function to pursue a target while avoiding obstacles
def pursue_target(uav, target, obstacles=None):
    """Generate pursuit behavior with obstacle avoidance.
    
    Args:
        uav: The UAV object
        target: The target object
        obstacles: List of obstacle objects to avoid
        
    Returns:
        np.array: Force vector for UAV movement
    """
    # Calculate direction and distance to target
    direction = target.position - uav.position
    distance = np.linalg.norm(direction)
    
    if distance > 0:
        direction = direction / distance  # Normalize
    
    # Check for nearby obstacles that require avoidance
    if obstacles and len(obstacles) > 0:
        # Enhanced obstacle avoidance using sensor data
        avoidance_force = np.zeros(2)
        for obstacle in obstacles:
            # Vector from UAV to obstacle
            to_obstacle = obstacle.position - uav.position
            obs_distance = np.linalg.norm(to_obstacle)
            
            # Only consider obstacles within detection range
            # Use UAV sensor range for safety margin
            safe_distance = obstacle.radius + CONFIG["sensor_range"] * 1.2
            
            if obs_distance < safe_distance:
                # Normalize direction to obstacle
                if obs_distance > 0:
                    to_obstacle = to_obstacle / obs_distance
                
                # Calculate how much the obstacle is in our path
                # Higher value means more directly in front
                in_path = np.dot(direction, to_obstacle)
                # Avoid any obstacle ahead
                if in_path > 0:
                    # Generate avoidance vector (perpendicular to obstacle direction)
                    # Use the cross product to get a consistent avoidance direction
                    avoid_dir = np.array([-to_obstacle[1], to_obstacle[0]])
                    
                    # Avoidance strength increases as we get closer
                    # Inverse square law for more responsive avoidance
                    avoid_strength = (1.0 / (obs_distance/safe_distance)**2) * 2.0
                    avoid_strength = min(avoid_strength, 3.0)  # Cap to prevent excessive force
                    
                    # Add to total avoidance force
                    avoidance_force += avoid_dir * avoid_strength * in_path
        
        # Add avoidance to direction
        if np.linalg.norm(avoidance_force) > 0:
            # Normalize avoidance force
            avoidance_force = avoidance_force / np.linalg.norm(avoidance_force)
            
            # Blend pursuit and avoidance equally for robust avoidance
            combined = direction * 0.5 + avoidance_force * 0.5
            
            # Re-normalize
            if np.linalg.norm(combined) > 0:
                direction = combined / np.linalg.norm(combined)
    
    # Calculate final force with approach factor
    # Slow down when approaching target for smoother positioning
    approach_factor = min(1.0, max(0.4, distance / 3.0))  # Between 0.4 and 1.0
    
    # Return final force vector
    force = direction * approach_factor
    return force

def min_distance_to_target(uavs, target):
    """Calculate minimum distance from any UAV to target"""
    min_dist = float('inf')
    for uav in uavs:
        dist = np.linalg.norm(uav.position - target.position)
        min_dist = min(min_dist, dist)
    return min_dist

def average_distance_to_target(uavs, target):
    """Calculate average distance from all UAVs to target"""
    total_dist = 0
    for uav in uavs:
        total_dist += np.linalg.norm(uav.position - target.position)
    return total_dist / len(uavs)

def is_target_encircled(uavs, target, threshold=1.8):
    """Check if UAVs have successfully encircled the target.
    
    Args:
        uavs: List of UAV objects
        target: Target object
        threshold: Maximum distance UAVs should be from target (in km)
        
    Returns:
        bool: True if target is successfully encircled
    """
    if len(uavs) < 3:
        return False  # Need at least 3 UAVs to form an encirclement
        
    # Calculate angles from target to each UAV (in radians)
    angles = []
    all_close_enough = True
    
    for uav in uavs:
        # Calculate distance and check if this UAV is close enough
        rel_pos = uav.position - target.position
        distance = np.linalg.norm(rel_pos)
        
        if distance > threshold:
            all_close_enough = False
        
        # Calculate angle in radians, normalize to [0, 2π)
        angle = np.arctan2(rel_pos[1], rel_pos[0]) % (2 * np.pi)
        angles.append(angle)
        
    # If any UAV is too far, encirclement fails
    if not all_close_enough:
        return False
        
    # Sort angles for gap analysis
    angles.sort()
    
    # Calculate gaps between consecutive UAVs (include wrap-around from last to first)
    gaps = []
    for i in range(len(angles)):
        next_idx = (i + 1) % len(angles)
        gap = (angles[next_idx] - angles[i]) % (2 * np.pi)
        gaps.append(gap)
        
    # For successful encirclement, no gap should be too large
    # With n UAVs, the ideal gap is 2π/n, we allow up to 1.8 times this size
    max_gap = max(gaps)
    max_allowed_gap = 1.8 * (2 * np.pi / len(uavs))
    
    return max_gap <= max_allowed_gap

def run_simple_demo(use_ahfsi=True, num_uavs=3, num_obstacles=3, max_steps=600, dynamic_obstacles=False):
    """Run a simplified military radar visualization with direct rendering"""
    # Print mission briefing
    print(Fore.CYAN + "\n===== UAV TACTICAL MISSION BRIEFING =====")
    print("Mission: Intercept and neutralize hostile target")
    print("Assets: " + Fore.GREEN + f"{num_uavs} UAV units" + Fore.CYAN)
    print("Terrain: " + Fore.YELLOW + f"{num_obstacles} hostile radar installations" + Fore.CYAN)
    print("Intelligence: Target is moving with evasive patterns")
    if use_ahfsi:
        print("System: " + Fore.GREEN + "AHFSI-enhanced swarm intelligence engaged" + Fore.CYAN)
    else:
        print("System: " + Fore.RED + "Standard tactical systems without AHFSI" + Fore.CYAN)
    print("=================================\n" + Style.RESET_ALL)
    
    # Update CONFIG with the provided parameters to ensure logging shows correct values
    CONFIG["num_uavs"] = num_uavs
    CONFIG["num_obstacles"] = num_obstacles
    CONFIG["mode"] = "demo"
    CONFIG["episodes"] = 1
    CONFIG["max_steps_per_episode"] = max_steps
    
    # Create environment with the three-stage operation capability
    env = Environment(num_uavs=num_uavs, num_obstacles=num_obstacles,
                      enable_ahfsi=use_ahfsi, dynamic_obstacles=dynamic_obstacles)
    
    # Initialize the simulation with pursuit behavior as fallback
    print(f"Initializing UAV simulation with {num_uavs} UAVs and {num_obstacles} obstacles")
    print("Using pursuit behavior as fallback for all UAVs")
    
    # No need to call env.reset() as it's already called inside Environment.__init__()
    
    # Create the figure with military-style aesthetics
    plt.style.use('dark_background')
    
    # Advanced performance optimizations for much better performance
    plt.rcParams['path.simplify'] = True
    plt.rcParams['path.simplify_threshold'] = 1.0  # Higher values = more simplification
    plt.rcParams['agg.path.chunksize'] = 20000  # Doubled for faster rendering
    plt.rcParams['figure.dpi'] = 100  # Lower DPI for faster rendering
    plt.rcParams['figure.autolayout'] = False  # Disable expensive autolayout
    plt.rcParams['savefig.bbox'] = 'tight'  # Faster for any potential saves
    plt.rcParams['figure.facecolor'] = '#0a0a0a'
    plt.rcParams['figure.autolayout'] = False  # Disable expensive autolayout
    plt.rcParams['toolbar'] = 'None'  # Disable toolbar for slightly better performance
    plt.rcParams['image.interpolation'] = 'nearest'  # Fastest interpolation
    
    # Create figure with multiple subplots for tactical display
    fig = plt.figure(figsize=(16, 16), dpi=100, facecolor='#0a0a0a')
    # Enable matplotlib to use the canvas buffer for faster rendering
    fig.canvas.draw()  # Initial draw to set up canvas
    
    # Store the background for advanced blitting optimization when supported
    try:
        background = fig.canvas.copy_from_bbox(fig.bbox)
        blitting_supported = True
    except (AttributeError, TypeError):
        background = None
        blitting_supported = False
    
    # Create optimized figure with pre-allocation of memory
    gs = gridspec.GridSpec(5, 6, figure=fig)
    
    # Target info panel - Reduced padding for tighter layout
    ax_target = fig.add_subplot(gs[0:2, 4:], facecolor='#1a1a2e')
    ax_target.set_title("Target Analysis", color='#00ffff', fontsize=12, pad=2)  # Reduced title padding
    ax_target.axis('off')
    
    # UAV status panel - Reduced padding for tighter layout
    ax_uav = fig.add_subplot(gs[2:4, 4:], facecolor='#1a1a2e')
    ax_uav.set_title("UAV Status", color='#00ffff', fontsize=12, pad=2)  # Reduced title padding
    ax_uav.axis('off')
    
    # Place mission status panel above main display
    ax_mission = fig.add_subplot(gs[0, 0:4], facecolor='#0a0a2b')
    ax_mission.set_title("Mission Status", color='#00ffff', fontsize=12, fontweight='bold')
    ax_mission.axis('off')
    # Add border to mission status panel for better definition
    mission_border = plt.Rectangle((0, 0), 1, 1, transform=ax_mission.transAxes, 
                                 fill=False, edgecolor='#00aaff', linewidth=2, alpha=0.8)
    ax_mission.add_patch(mission_border)  # Actually add the border to the plot
    
    # Adjust main display to accommodate the mission panel
    ax = fig.add_subplot(gs[1:4, :4])
    ax.set_facecolor('#0a3b0a')  # Dark green radar background
    
    # Set limits and grid
    width = CONFIG.get("scenario_width", 15.0)
    height = CONFIG.get("scenario_height", 15.0)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    # Create standard grid and axis setup
    ax.grid(True, color='#00ff00', alpha=0.2, linestyle='-')
    ax.set_title("Military Tactical UAV Simulation", color='#00ff00', fontsize=14)
    
    # Move X and Y labels further from the axes to avoid overlap with numbers
    ax.set_xlabel("X Position (km)", color='white', labelpad=15)  # Increased padding
    ax.set_ylabel("Y Position (km)", color='white', labelpad=15)  # Increased padding
    
    # Start fresh - remove all default ticks first
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Now manually add custom tick marks and labels with better visibility
    # Add proper axis numbers at the edges of the plot (white) - every 1km, non-bold
    for x in range(0, int(width)+1, 1):
        ax.text(x, -0.5, f"{x}", fontsize=9, color='white', alpha=1.0, ha='center')
    for y in range(0, int(height)+1, 1):
        # Position y-axis labels closer to the axis and ensure they're not tilted
        ax.text(-0.35, y, f"{y}", fontsize=9, color='white', alpha=1.0, va='center', ha='center', rotation=0)  # Centered, closer to axis, no rotation
    
    # Remove ALL minor ticks completely
    ax.tick_params(axis='both', which='minor', length=0)
        
    # Add radar circles
    center_x, center_y = width/2, height/2
    for radius in [2, 4, 6, 8, 10]:
        circle = Circle((center_x, center_y), radius, fill=False, 
                       edgecolor='#00aa00', alpha=0.3, linestyle='--')
        ax.add_patch(circle)
        
    # Add compass markers (N,S,E,W)
    compass_points = [
        (center_x, center_y + 10.2, 'N'),
        (center_x, center_y - 10.2, 'S'),
        (center_x + 10.2, center_y, 'E'),
        (center_x - 10.2, center_y, 'W')
    ]
    
    for x, y, label in compass_points:
        if 0 <= x <= width and 0 <= y <= height:  # Only add if within bounds
            text = ax.text(x, y, label, fontsize=10, color='#00ffaa', ha='center', va='center',
                        path_effects=[path_effects.withStroke(linewidth=2, foreground='#003300')])
            
    # Coordinate grid labels removed as requested
    
    # Create enhanced UAV shapes as quadcopters (drones with four rotors)
    uav_bodies = []      # Main UAV body
    uav_arms = []        # Arms connecting body to rotors
    uav_rotors = []      # List of rotor sets for each UAV
    uav_sensors = []     # Sensor range circles
    uav_centers = []     # Completely hide all axis elements but keep the grid
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # This ensures no ticks or tick labels can possibly remain visible
    sensor_range = CONFIG.get('sensor_range', 0.7)  # Get sensor range from config
    
    # Color scheme - modern military-tech look
    body_color = '#101820'       # Almost black body
    body_edge = '#00ff66'        # Bright green edge (tech look)
    rotor_color = '#2c3e50'      # Dark blue-gray for rotors
    center_color = '#00ff99'     # Bright cyan center highlight
    arm_color = '#30cfd0'        # Teal for arms
    
    for i in range(num_uavs):
        # Create main body of quadcopter
        body = Circle(
            (0, 0),  # Will be updated with actual position
            0.15,    # Size of center body
            facecolor=body_color,
            edgecolor=body_edge,
            linewidth=1.5,
            path_effects=[path_effects.withStroke(linewidth=3, foreground='#003311')],
            label='UAVs' if i == 0 else '',  # Only label first one for legend
            zorder=12
        )
        ax.add_patch(body)
        uav_bodies.append(body)
        
        # Add center highlight (LED-like effect)
        center = Circle(
            (0, 0),  # Will be updated with position
            0.05,    # Small center highlight
            facecolor=center_color,
            alpha=0.8,
            zorder=13
        )
        ax.add_patch(center)
        uav_centers.append(center)
        
        # Create arms for each rotor (4 arms)
        arm_set = []
        for j in range(4):  # 4 arms
            arm, = ax.plot(
                [0, 0], [0, 0],  # Will be updated
                color=arm_color,
                linewidth=3.0,
                solid_capstyle='round',
                path_effects=[path_effects.withStroke(linewidth=4, foreground='#003311')],
                zorder=11
            )
            arm_set.append(arm)
        uav_arms.append(arm_set)
        
        # Create four rotors for this UAV
        rotor_set = []
        for j in range(4):  # 4 rotors
            # Alternate colors for front/back rotors for better orientation
            r_color = rotor_color if j in [1, 2] else '#4682B4'  # Back rotors use rotor_color, front rotors are blue
            
            rotor = Circle(
                (0, 0),  # Will be updated with positions
                0.075,   # Size of rotors
                facecolor=r_color,
                edgecolor='#a0a0a0',
                linewidth=1.5,
                alpha=0.9,
                zorder=12
            )
            ax.add_patch(rotor)
            rotor_set.append(rotor)
        uav_rotors.append(rotor_set)
        
        # Add enhanced sensor range circle with radar-like appearance
        sensor = Circle(
            (0, 0),  # Will be updated with UAV position
            sensor_range,  # Radius based on sensor range
            fill=False,
            edgecolor='#00ff99',  # Brighter green for better visibility
            linestyle=(0, (3, 2)),  # Custom dashed pattern for modern radar look
            linewidth=1.5,
            alpha=0.5,
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground='#003311', alpha=0.3)],
            zorder=5
        )
        ax.add_patch(sensor)
        uav_sensors.append(sensor)
    
    # Create target visualization
    target_scatter = ax.scatter(
        [], [], 
        s=150,  # Size
        marker='*',  # Star marker for target
        color='red',
        edgecolor='#ffff00',
        linewidth=1.5,
        label='Target',
        zorder=10
    )
    
    # Add capture zone around target - visible capture radius
    capture_radius = CONFIG.get('roundup_strategy_threshold', 3.0)
    capture_circle = Circle(
        (0, 0),  # Position updated in animation
        capture_radius,
        fill=False, 
        edgecolor='red',
        linestyle='--',
        alpha=0.7,
        zorder=5
    )
    ax.add_patch(capture_circle)
    
    # Add target info text to target panel with boxed background - better positioned
    target_info_text = ax_target.text(
        0.5, 0.6, 
        "Target Information\nLoading...",
        transform=ax_target.transAxes,
        fontsize=12,
        color='#00ffff',
        horizontalalignment='center',
        verticalalignment='center',
        family='monospace',
        bbox=dict(facecolor='#000022', alpha=0.8, edgecolor='#0088ff', pad=10)
    )
    
    # Create obstacles with radar lines
    obstacle_patches = []
    obstacle_radar_rings = []
    obstacle_scan_angles = []  # Store current angles for scanning effect
    
    for obstacle in env.obstacles:
        # Obstacles displayed at their physical size
        obstacle_patch = Circle(
            (obstacle.position[0], obstacle.position[1]),
            obstacle.radius,  # Actual physical size
            facecolor='#884400',
            alpha=0.6,
            edgecolor='#aa6600',
            linewidth=1.5
        )
        ax.add_patch(obstacle_patch)
        obstacle_patches.append(obstacle_patch)
        
        # Add radar scan line - initially points to angle 0
        radar_line, = ax.plot(
            [obstacle.position[0], obstacle.position[0] + obstacle.radius], 
            [obstacle.position[1], obstacle.position[1]],
            color='#ffaa00', linewidth=1.0, alpha=0.7, zorder=5
        )
        obstacle_radar_rings.append(radar_line)
        obstacle_scan_angles.append(0)  # Start at angle 0
    
    # Mission status text
    mission_text = ax_mission.text(
        0.5, 0.5, 
        "MISSION: ACTIVE\nSTATUS: TARGET ACQUISITION", 
        transform=ax_mission.transAxes,
        fontsize=10,
        color='#00ffff',
        horizontalalignment='center',
        verticalalignment='center',
        family='monospace',
        bbox=dict(facecolor='#001133', alpha=0.8, edgecolor='#0066aa', pad=10)
    )
    
    # UAV status text in the UAV panel with boxed background - better positioned
    uav_status_text = ax_uav.text(
        0.5, 0.6, 
        "UAV Status\nLoading...",
        transform=ax_uav.transAxes,
        fontsize=12,
        color='#00ffff',
        horizontalalignment='center',
        verticalalignment='center',
        family='monospace',
        bbox=dict(facecolor='#000033', alpha=0.8, edgecolor='#0088ff', pad=10)
    )
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.7)
    
    # Collect all text elements for efficient updating
    text_elements = [mission_text, uav_status_text, target_info_text]
    
    # Load trained models if available
    checkpoint_dir = "checkpoints/ahfsi" if use_ahfsi else "checkpoints/standard"
    
    # Create a specialized UAV controller class
    class UAVModelController:
        """A wrapper class for handling UAV model loading and prediction"""
        
        def __init__(self, model_path, state_dim, action_dim):
            """Initialize a UAV controller
            
            Args:
                model_path: Path to the saved model file
                state_dim: Dimension of the state/observation space
                action_dim: Dimension of the action space
            """
            self.model_path = model_path
            self.model_loaded = False
            self.state_dim = state_dim
            self.action_dim = action_dim
            
            # Load model weights into ActorNetwork instead of full model
            if os.path.exists(model_path):
                # Determine actor name from filename
                actor_name = os.path.splitext(os.path.basename(model_path))[0]
                # Instantiate ActorNetwork and load weights
                network = ActorNetwork(self.state_dim, self.action_dim, actor_name)
                try:
                    network.model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    network.target_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    self.model = network.model
                    self.model_loaded = True
                    print(f"Successfully loaded weights for {actor_name} from {model_path}")
                except Exception as e:
                    print(f"Error loading weights: {e}")
                    self.model_loaded = False
            else:
                print(f"Model weights file not found: {model_path}")
                self.model_loaded = False
        
        def predict(self, observation):
            """Make a prediction using the loaded model
            
            Args:
                observation: UAV observation/state
                
            Returns:
                Predicted action or None if prediction fails
            """
            if not self.model_loaded:
                return None
                
            try:
                # Format observation as tensor
                import tensorflow as tf
                import numpy as np
                
                # Ensure observation has the right shape
                if len(observation.shape) == 1:
                    # Add batch dimension if missing
                    observation = np.expand_dims(observation, axis=0)
                    
                # Convert to tensor with correct type
                obs_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
                
                # Get prediction
                tf_output = self.model(obs_tensor, training=False)
                # Convert output tensor to numpy array
                if hasattr(tf_output, 'numpy'):
                    prediction = tf_output.numpy()
                else:
                    prediction = np.array(tf_output)
                
                # Extract action based on prediction format
                if isinstance(prediction, list):
                    action = prediction[0]
                elif hasattr(prediction, 'numpy'):
                    # TensorFlow tensor
                    action = prediction.numpy()[0]
                else:
                    # Numpy array
                    action = prediction[0]
                    
                return action
            except Exception as e:
                print(f"Prediction error: {e}")
                return None
    
    # Load models for each UAV
    controllers = []
    if os.path.exists(checkpoint_dir):
        print(f"Loading models from {checkpoint_dir}")
        
        for i in range(num_uavs):
            # Get state dimension for current UAV
            state_dim = env.uavs[i].get_observation().shape[0]
            
            # Create model path based on checkpoint directory and UAV index
            actor_name = f"{('ahfsi_' if use_ahfsi else '')}actor_{i}"
            model_path = os.path.join(checkpoint_dir, f"{actor_name}.h5")
            
            # Create controller for this UAV
            controller = UAVModelController(model_path, state_dim, 2)
            controllers.append(controller)
            
            # Attach controller to UAV
            env.uavs[i].controller = controller
            env.uavs[i].has_controller = controller.model_loaded
            
            # Also connect to AHFSI controller if available
            if use_ahfsi and hasattr(env.uavs[i], 'ahfsi_controller'):
                env.uavs[i].ahfsi_controller.controller = controller
                if controller.model_loaded:
                    print(f"Connected trained model to UAV {i}'s AHFSI controller")
    else:
        print(f"Warning: No trained models found in {checkpoint_dir}")
        print("Running with random actions")
    
    # Single-run logging of model usage per UAV
    for i, uav in enumerate(env.uavs):
        if uav.has_controller:
            print(f"UAV {i} using trained model")
        else:
            print(f"UAV {i} using fallback behavior")

    # Make window visible
    # Make window visible
    plt.ion()  # Interactive mode on
    plt.tight_layout()
    plt.show()
    
    # Display simulation information
    print(f"{Fore.CYAN}[System] Simulation initialized with trained models{Style.RESET_ALL}")
    
    # Simulation variables
    actors = []
    running = True
    cumulative_reward = 0.0
    step = 0
    collisions = 0
    vis_time = 0.0  # Initialize visualization time tracking
    
    # Record start time for mission tracking
    start_time = time.time()
    print("Starting simulation... Press Ctrl+C to stop")
    
    try:
        while running:
            loop_start = time.perf_counter()

            # Get actions from trained policy or random if no policy available
            action_start = time.perf_counter()

            # Agent decision process (calculate actions)
            actions = []

            for i, uav in enumerate(env.uavs):
                # Get action from trained model or use fallback when necessary
                if hasattr(uav, 'has_controller') and uav.has_controller:
                    # Get observation from the UAV
                    observation = uav.get_observation()
                    
                    # Predict action using the controller
                    try:
                        # Get prediction from the controller
                        predicted_action = uav.controller.predict(observation)
                        
                        if predicted_action is not None:
                            # Successfully got a prediction from the model
                            action = predicted_action
                            
                            # Apply scaling for smooth control
                            max_force = 0.8  # Maximum force for UAV control
                            action = action * max_force
                        else:
                            # Prediction failed, use pursuit behavior
                            action = pursue_target(uav, env.target, env.obstacles)
                    except Exception as e:
                        # Handle any errors by falling back to basic behavior
                        print(f"Model error for UAV {i}: {e}, using fallback")
                        action = pursue_target(uav, env.target, env.obstacles)
                else:
                    # No controller available, use basic pursuit behavior
                    action = pursue_target(uav, env.target, env.obstacles)

                        
                actions.append(action)

            action_time = (time.time() - action_start) * 1000  # Convert to ms

            # Step environment
            env_step_start = time.perf_counter()
            _, rewards, dones, info = env.step(actions)
            env_step_end = time.perf_counter()
            env_step_time = env_step_end - env_step_start  # Calculate time for environment step
            cumulative_reward += np.mean(rewards)
            step += 1

            for uav in env.uavs:
                if hasattr(uav, 'collision_detected') and uav.collision_detected:
                    collisions += 1

            # PERFORMANCE OPTIMIZATION: Visualization update with performance tracking
            vis_start = time.time()

            # Adaptive frame skipping for better performance
            if vis_time > 0.05 and step % 2 == 0:  # Skip every other frame if visualization is taking >50ms
                # Only update text elements which are cheap to render
                if step % 5 == 0:
                    elapsed = time.time() - start_time
                    hours, remainder = divmod(int(elapsed), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    mission_time = f"MISSION TIME: {hours:02d}:{minutes:02d}:{seconds:02d}"
                    mission_text.set_text(f"MISSION: ACTIVE\nSTATUS: {info.get('task_state', 'tracking').upper()}\n{mission_time}")

                # Skip the rest of the rendering for this frame
                vis_time = 0  # Reset for next calculation

                # Move to next iteration of the loop
                continue
            
            # Check if we've reached the maximum steps or success conditions
            if step >= max_steps:
                print(f"Simulation reached maximum steps ({max_steps}), terminating")
                running = False
                break
            
            # Check if target has been captured based on environment's info state
            # This ensures the visualization and environment are in sync
            if info.get('success', False):
                # Only print this message once per capture
                if not hasattr(env, 'capture_displayed') or not env.capture_displayed:
                    print(f"\n{Fore.GREEN}TARGET CAPTURED! ROUNDUP COMPLETE!{Style.RESET_ALL}")
                    env.capture_displayed = True
                    
                # Allow a few more steps to show the capture
                if step % 5 == 0 and step > 50:
                    running = False
                    break

            # Only update visualization on every frame for smoother animation
            # This provides better visual smoothness at the cost of some performance
            if step % 1 == 0:
                # UAVs - update military drone shapes and sensor ranges
                for i, uav in enumerate(env.uavs):
                    # Get UAV position and orientation
                    pos = uav.position
                    if hasattr(uav, 'orientation'):
                        orientation = uav.orientation
                    else:
                        # If no orientation attribute, calculate from velocity
                        if hasattr(uav, 'velocity') and np.linalg.norm(uav.velocity) > 0.01:
                            orientation = np.arctan2(uav.velocity[1], uav.velocity[0])
                        else:
                            # Default orientation (facing right)
                            orientation = 0

                    # Update sensor range circle
                    uav_sensors[i].center = (pos[0], pos[1])

                    # Update the quadcopter body and highlight position
                    uav_bodies[i].center = (pos[0], pos[1])
                    uav_centers[i].center = (pos[0], pos[1])

                    # Calculate rotor positions - in square formation around body
                    # The four rotors are positioned along axes rotated by the orientation
                    # Front-right, back-right, back-left, front-left
                    arm_length = 0.3  # Slightly longer arms for better proportions

                    # Adjust rotor angles based on orientation
                    rotor_angles = [
                        orientation + np.pi/4,     # Front-right (45 degrees)
                        orientation + 3*np.pi/4,   # Back-right (135 degrees)
                        orientation + 5*np.pi/4,   # Back-left (225 degrees)
                        orientation + 7*np.pi/4    # Front-left (315 degrees)
                    ]

                    # Update the four rotors and arms
                    for j, angle in enumerate(rotor_angles):
                        # Calculate position of this rotor
                        rotor_x = pos[0] + arm_length * np.cos(angle)
                        rotor_y = pos[1] + arm_length * np.sin(angle)

                        # Update rotor position
                        uav_rotors[i][j].center = (rotor_x, rotor_y)

                        # Update arm connecting body to rotor
                        uav_arms[i][j].set_data(
                            [pos[0], rotor_x],
                            [pos[1], rotor_y]
                        )

                # PERFORMANCE OPTIMIZATION: Adaptive visualization frequency
                # Only update slower-changing elements at reduced frequency - adjusts based on performance
                vis_update_interval = 3  # Update every 3 frames by default, adjusted dynamically below

                # If visualization is taking too long, reduce update frequency
                if hasattr(locals(), 'vis_time') and vis_time > 0.05:  # If last vis_time > 50ms
                    vis_update_interval = 6  # Reduce update frequency when performance is struggling
                if hasattr(locals(), 'vis_time') and vis_time > 0.1:  # If last vis_time > 100ms
                    vis_update_interval = 12  # Significantly reduce updates for very slow performance

                if step % vis_update_interval == 0:
                    # Target
                    target_position = env.target.position
                    target_scatter.set_offsets([target_position])

                    # Capture circle
                    capture_circle.center = (target_position[0], target_position[1])

                    # Obstacles
                    for i, obstacle in enumerate(env.obstacles):
                        obstacle_position = obstacle.position
                        obstacle_patches[i].center = (obstacle_position[0], obstacle_position[1])

                        # Update radar scan line with rotating effect
                        obstacle_scan_angles[i] = (obstacle_scan_angles[i] + 0.1) % (2 * np.pi)
                        angle = obstacle_scan_angles[i]

                        # Calculate endpoint of radar line
                        scan_radius = obstacle.radius  # Use actual physical size
                        end_x = obstacle.position[0] + scan_radius * np.cos(angle)
                        end_y = obstacle.position[1] + scan_radius * np.sin(angle)

                        # Update the line data
                        obstacle_radar_rings[i].set_data(
                            [obstacle.position[0], end_x],
                            [obstacle.position[1], end_y]
                        )

                # Format the mission time
                elapsed = time.time() - start_time
                hours, remainder = divmod(int(elapsed), 3600)
                minutes, seconds = divmod(remainder, 60)
                mission_time = f"MISSION TIME: {hours:02d}:{minutes:02d}:{seconds:02d}"

                # Get mission phase from environment state
                phase_distances = []
                for uav in env.uavs:
                    distance = np.linalg.norm(uav.position - env.target.position)
                    phase_distances.append(distance)
                
                avg_distance = sum(phase_distances) / len(phase_distances)
                min_distance = min(phase_distances) if phase_distances else float('inf')
                max_distance = max(phase_distances) if phase_distances else float('inf')
                
                # Get basic status information for display
                task_state = info.get('task_state', 'tracking')
                
                # Only log state occasionally to avoid console flooding
                if step % 50 == 0:
                    # Calculate distances for status display
                    distances = []
                    for uav in env.uavs:
                        distance = np.linalg.norm(uav.position - env.target.position)
                        distances.append(distance)
                    avg_distance = sum(distances) / len(distances) if distances else 0.0
                    min_distance = min(distances) if distances else 0.0
                    
                    # Log basic target tracking information
                    print(f"Status: {task_state.upper()}, avg_dist={avg_distance:.2f}, min_dist={min_distance:.2f}")
                
                # Update status display with current mission state
                status_msg = f"MISSION: ACTIVE\nSTATUS: {task_state.upper()}\n{mission_time}"
                mission_text.set_text(status_msg)

                # Update UAV status - detailed info for each UAV
                uav_status = "UAV STATUS:\n" + "-"*20 + "\n"
                for i, uav in enumerate(env.uavs):
                    pos = uav.position
                    vel = uav.velocity
                    speed = np.linalg.norm(vel)
                    # Calculate heading in degrees (0° is East, 90° is North)
                    heading = np.degrees(np.arctan2(vel[1], vel[0])) % 360
                    distance_to_target = np.linalg.norm(pos - env.target.position)

                    # Color code based on distance to target
                    if distance_to_target < 2:
                        color_code = '^g'  # Green for close to target
                    elif distance_to_target < 4:
                        color_code = '^y'  # Yellow for medium distance
                    else:
                        color_code = '^r'  # Red for far from target

                    # Add UAV info line with fixed width formatting for proper alignment
                    uav_status += f"UAV-{i+1}: Pos[{pos[0]:5.1f},{pos[1]:5.1f}] "
                    uav_status += f"Hdg:{heading:3.0f}° Spd:{speed:4.1f} {color_code}\n"

                # Update UAV status text
                uav_status_text.set_text(uav_status)

                # Update target information
                target_pos = env.target.position
                target_vel = env.target.velocity
                target_speed = np.linalg.norm(target_vel)
                target_heading = np.degrees(np.arctan2(target_vel[1], target_vel[0])) % 360

                # Count UAVs in roundup position
                uavs_in_range = 0
                min_distance = float('inf')
                for uav in env.uavs:
                    dist = np.linalg.norm(uav.position - target_pos)
                    min_distance = min(min_distance, dist)
                    if dist < capture_radius:
                        uavs_in_range += 1

                target_info = f"TARGET INFORMATION:\n" + "-"*20 + "\n"
                target_info += f"Position: [{target_pos[0]:.1f}, {target_pos[1]:.1f}]\n"
                target_info += f"Heading: {target_heading:.0f}° Speed: {target_speed:.1f}\n"
                target_info += f"UAVs in range: {uavs_in_range}/{num_uavs}\n"
                target_info += f"Closest UAV: {min_distance:.2f} km\n"

                # Add threat level based on UAVs in range
                if uavs_in_range >= 3:
                    target_info += "\nTHREAT LEVEL: *CRITICAL*"
                elif uavs_in_range >= 2:
                    target_info += "\nTHREAT LEVEL: HIGH"
                elif uavs_in_range >= 1:
                    target_info += "\nTHREAT LEVEL: MODERATE"
                else:
                    target_info += "\nTHREAT LEVEL: LOW"

                target_info_text.set_text(target_info)

                # Check for completion
                done_flag = dones if isinstance(dones, bool) else any(dones)
                if done_flag:
                    # Complete success message with mission time
                    success_message = f"MISSION: SUCCESS\nTARGET CAPTURED\n{mission_time}"
                    mission_text.set_text(success_message)
                    mission_text.set_color('#ff3333')  # Bright red for mission success
                    mission_text.set_bbox(dict(facecolor='#300000', alpha=1.0, edgecolor='#ff0000', pad=5))

                    print(f"\n{Fore.GREEN}TARGET CAPTURED! ROUNDUP COMPLETE!{Style.RESET_ALL}")

                # Advanced rendering pipeline optimization with adaptive refresh
                # Optimize refresh frequency based on current performance
                if step % max(1, int(5 * (1 + min(vis_time, 0.2) * 10))) == 0:  # Dynamic adjustment

                    # Use optimized double buffering with background restoration when available
                    if blitting_supported and background is not None:
                        try:
                            # Step 1: Restore background - much faster than redrawing everything
                            try:
                                fig.canvas.restore_region(background)
                            except (AttributeError, ValueError):
                                # Fall back if restore fails
                                pass
                                
                            # Step 2: Redraw just the updated artists
                            for i, uav in enumerate(env.uavs):
                                ax.draw_artist(uav_bodies[i])
                                ax.draw_artist(uav_centers[i])
                                ax.draw_artist(uav_sensors[i])

                            # Step 3: Draw target
                            ax.draw_artist(target_scatter)
                                
                            # Step 4: Only draw detail elements when performance allows it
                            if vis_time < 0.1:  # Only when we have rendering headroom
                                # Draw text elements
                                ax.draw_artist(mission_text)
                                ax.draw_artist(uav_status_text)
                                ax.draw_artist(target_info_text)
                                    
                                # Draw UAV details (rotors and arms)
                                for i in range(len(env.uavs)):
                                    for rotor in uav_rotors[i]:
                                        ax.draw_artist(rotor)
                                    for arm in uav_arms[i]:
                                        ax.draw_artist(arm)

                            # Step 5: Blit the updated region to display
                            fig.canvas.blit(ax.bbox)
                        except (AttributeError, ValueError, TypeError):
                            # Fall back to optimized draw_idle if blitting fails
                            fig.canvas.draw_idle()
                    else:
                        # Fall back for backends that don't support blitting
                        fig.canvas.draw_idle()  # Still faster than full draw()

                # Process GUI events with throttling for better performance
                try:
                    fig.canvas.flush_events()
                except (NotImplementedError, AttributeError):
                    pass  # Some backends don't support this

                # Calculate visualization time for profiling
                vis_end = time.perf_counter()
                vis_time = vis_end - vis_start
                
                # Print profiling info every 2 steps
            if step % 50 == 0:
                print(f"[PROFILE] Step {step}: Action: {action_time:.2f} ms | Env: {env_step_time*1000:.2f} ms | Vis: {vis_time*1000:.2f} ms")
            
            # Print occasionally to avoid console flooding
            if step % 5 == 0:
                print(f"Step {step}, Reward: {np.mean(rewards):.4f}, Total: {cumulative_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Final mission statistics
    elapsed_time = time.time() - start_time
    print(f"\n{Fore.CYAN}===== MISSION SUMMARY ====={Style.RESET_ALL}")
    print(f"Duration: {elapsed_time:.1f} seconds ({step} simulation steps)")
    print(f"Performance Score: {cumulative_reward:.2f}")
    print(f"UAV Collisions: {collisions}")
    
    # Evaluate mission success based on target encirclement
    try:
        # Use the environment's target encirclement check for final success evaluation
        target_encircled = env._is_target_encircled()
        target_visibility = env._evaluate_target_visibility()
        visible_count = sum(target_visibility)
        min_distance = min([np.linalg.norm(uav.position - env.target.position) for uav in env.uavs])
        
        # Success only if target is properly surrounded according to roundup strategy
        roundup_success = target_encircled and visible_count >= 3 and min_distance < 2.5
        
        print(f"Mission Status: {Fore.GREEN if roundup_success else Fore.RED}" + 
              ("SUCCESS - Target captured in roundup formation" if roundup_success else "INCOMPLETE - Roundup formation not achieved") + 
              f"{Style.RESET_ALL}")
        if roundup_success:
            print(f"UAVs in roundup formation: {visible_count} | Min Distance: {min_distance:.2f} km")
    except:
        # Fallback if we can't evaluate encirclement
        print(f"Mission Status: {Fore.YELLOW}UNKNOWN - Could not evaluate roundup completion{Style.RESET_ALL}")
    
    print(f"{Fore.CYAN}=========================={Style.RESET_ALL}")
    
    # Keep window open until closed by user
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simple Military Radar Visualization")
    parser.add_argument("--ahfsi", action="store_true", default=True, 
                      help="Use AHFSI-enhanced models (default: True)")
    parser.add_argument("--no-ahfsi", action="store_false", dest="ahfsi",
                      help="Use standard models without AHFSI")
    parser.add_argument("--uavs", type=int, default=3,
                      help="Number of UAVs in simulation (default: 3)")
    parser.add_argument("--obstacles", type=int, default=3,
                      help="Number of obstacles in simulation (default: 3)")
    parser.add_argument("--dynamic-obstacles", action="store_true", dest="dynamic_obstacles",
                        help="Enable dynamic obstacle movement (default: disabled)")

    args = parser.parse_args()
    
    run_simple_demo(
        use_ahfsi=args.ahfsi,
        num_uavs=args.uavs,
        num_obstacles=args.obstacles,
        dynamic_obstacles=args.dynamic_obstacles
    )
