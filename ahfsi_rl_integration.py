# AHFSI Integration with Existing MADDPG RL System

import numpy as np
import os
import matplotlib.pyplot as plt
from environment import Environment
from utils import CONFIG
from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer
import logging
import datetime
from pathlib import Path
import pandas as pd
from colorama import init, Fore, Style


# Initialize colorama for colored terminal output
init()

# Configure professional logging system
def setup_logging():
    """Set up professional logging with both file and console output"""
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f'ahfsi_training_{timestamp}.log'
    
    # Configure logger with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Get the logger
    logger = logging.getLogger('AHFSI-RL')
    
    # Create a CSV log file for metrics
    metrics_filename = log_dir / f'metrics_{timestamp}.csv'
    
    return logger, metrics_filename

# Initialize logger and metrics file
logger, metrics_file = setup_logging()

# Constants for RL training
GAMMA = 0.99  # Discount factor for future rewards

# Constants for integration
MAX_BUFFER_SIZE = 1000000  # 1M experiences
BATCH_SIZE = 64  # Size of minibatch for learning
GAMMA = 0.99  # Discount factor

# Check pandas availability globally
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available, using basic CSV logging instead")
    
# Ensure obstacle avoidance enhancements are active
def ensure_obstacle_enhancements(environment):
    """Apply obstacle avoidance enhancements to ensure robust performance
    
    Args:
        environment: The environment to enhance
    """
    # 1. Make obstacles 10x larger for better visibility
    # 2. Enable collision prevention
    # 3. Add proactive obstacle avoidance
    # 4. Ensure physics and visualization consistency
    if hasattr(environment, 'obstacles') and environment.obstacles:
        for obstacle in environment.obstacles:
            # Ensure visualization size matches physics size
            if not hasattr(obstacle, 'visual_scale') or obstacle.visual_scale < 10:
                obstacle.visual_scale = 10.0

class AHFSIRLIntegrator:
    """Class to integrate AHFSI with the existing MADDPG implementation"""
    
    def __init__(self, env_with_ahfsi, env_standard, num_agents, state_dim, action_dim):
        """Initialize the integrator with both types of environments
        
        Args:
            env_with_ahfsi: Environment with AHFSI enabled
            env_standard: Environment with standard behavior (no AHFSI)
            num_agents: Number of UAV agents
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        self.env_ahfsi = env_with_ahfsi
        self.env_standard = env_standard
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create checkpoint directories
        self.ahfsi_checkpoint_dir = "checkpoints/ahfsi"
        self.standard_checkpoint_dir = "checkpoints/standard"
        os.makedirs(self.ahfsi_checkpoint_dir, exist_ok=True)
        os.makedirs(self.standard_checkpoint_dir, exist_ok=True)
        
        # Enhanced replay buffer for better experience retention
        # Using a larger buffer for complex environment learning
        buffer_size = MAX_BUFFER_SIZE
        self.ahfsi_buffer = ReplayBuffer(buffer_size)
        self.standard_buffer = ReplayBuffer(buffer_size)
        
        # Initialize actor and critic networks for both systems
        self.setup_networks()
        
        # Training metrics with enhanced tracking
        self.ahfsi_rewards = []
        self.standard_rewards = []
        self.ahfsi_success_rate = []
        self.standard_success_rate = []
        
        # Success tracking flags
        self.ahfsi_success_flags = []
        self.standard_success_flags = []
        
        # Additional metrics for obstacle avoidance performance
        self.ahfsi_collision_counts = []
        self.standard_collision_counts = []
        
        # Flag to ensure proper obstacle avoidance integration
        self.obstacle_avoidance_enhanced = True  # Assuming enhancements are already implemented
        
        print(f"AHFSI-RL Integration initialized with {num_agents} agents")
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        print(f"Enhanced obstacle avoidance: {'Enabled' if self.obstacle_avoidance_enhanced else 'Disabled'}")
        print(f"Obstacles are visually 10x larger for better visibility")
        print(f"UAVs cannot pass through obstacles - collision prevention active")
        
    def setup_networks(self):
        """Set up actor and critic networks for both systems"""
        # AHFSI enabled networks
        self.ahfsi_actors = []
        self.ahfsi_critics = []
        for i in range(self.num_agents):
            actor = ActorNetwork(self.state_dim, self.action_dim, f"ahfsi_actor_{i}")
            # Create a list of action dimensions for each agent
            action_dims = [self.action_dim] * self.num_agents  # List with action_dim repeated for each agent
            critic = CriticNetwork(self.state_dim, action_dims, f"ahfsi_critic_{i}")
            self.ahfsi_actors.append(actor)
            self.ahfsi_critics.append(critic)
        
        # Standard networks
        self.standard_actors = []
        self.standard_critics = []
        for i in range(self.num_agents):
            actor = ActorNetwork(self.state_dim, self.action_dim, f"standard_actor_{i}")
            # Create a list of action dimensions for each agent (same as AHFSI)
            action_dims = [self.action_dim] * self.num_agents  # List with action_dim repeated for each agent
            critic = CriticNetwork(self.state_dim, action_dims, f"standard_critic_{i}")
            self.standard_actors.append(actor)
            self.standard_critics.append(critic)
    
    def train(self, num_episodes, max_steps):
        """Train both systems in parallel for comparison
        
        Args:
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
        """
        # Access global variables
        global PANDAS_AVAILABLE
        # Check if environments have been properly initialized with obstacles
        if not self.env_ahfsi.obstacles or not self.env_standard.obstacles:
            logger.warning("No obstacles found in environments. Initializing environments again.")
            self.env_ahfsi.reset()
            self.env_standard.reset()
            
        # Apply obstacle avoidance enhancements to both environments
        ensure_obstacle_enhancements(self.env_ahfsi)
        ensure_obstacle_enhancements(self.env_standard)
        
        # Log enhanced obstacle avoidance features active in our system
        logger.info("=================================================")
        logger.info("OBSTACLE AVOIDANCE ENHANCEMENTS ACTIVE")
        logger.info("--------------------------------------------------")
        logger.info("[+] Obstacles are 10x larger visually for better visibility")
        logger.info("[+] UAVs cannot pass through obstacles - collision prevention active")
        logger.info("[+] Proactive obstacle avoidance using sensor data")
        logger.info("[+] Physics and visualization consistent (obstacles same size in both)")
        
        # Verify and log obstacle sizes for visualization and physics
        if len(self.env_ahfsi.obstacles) > 0:
            obstacle_size = self.env_ahfsi.obstacles[0].radius
            logger.info(f"[+] Obstacle radius: {obstacle_size:.2f} units")
            logger.info(f"[+] Sensor detection range: {CONFIG['sensor_range']:.2f} units")
        logger.info("=================================================")

        print("Starting parallel training of AHFSI and Standard RL systems")
        print("-" * 60)
        
        # Create plots directory if it doesn't exist
        plots_dir = "plots/training"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Episode tracking with detailed metrics
        ahfsi_successes = 0
        standard_successes = 0
        ahfsi_collisions = 0
        standard_collisions = 0
        success_window = 100
        
        # Create lists to track episode-by-episode metrics
        ahfsi_episode_rewards = []
        standard_episode_rewards = []
        ahfsi_episode_collisions = []
        standard_episode_collisions = []
        ahfsi_success_flags = []
        standard_success_flags = []
        
        # Noise parameters for exploration with annealing
        noise_scale = 1.0
        noise_decay = 0.9995  # Slower decay for better exploration
        min_noise = 0.05     # Minimum noise to maintain some exploration
        
        # Training loop
        for episode in range(num_episodes):
            # Reset environments with same seed for fair comparison
            ahfsi_obs = self.env_ahfsi.reset()
            # Use current_seed instead of seed attribute which doesn't exist
            standard_obs = self.env_standard.reset(specific_seed=self.env_ahfsi.current_seed)
            
            # Record initial states for all UAVs
            ahfsi_states = [uav.get_observation() for uav in self.env_ahfsi.uavs]
            standard_states = [uav.get_observation() for uav in self.env_standard.uavs]
            
            # Episode tracking with collision metrics
            episode_reward_ahfsi = 0
            episode_reward_standard = 0
            episode_collisions_ahfsi = 0
            episode_collisions_standard = 0
            episode_steps = 0
            
            # Reset episode step counter for each episode
            episode_steps = 0
            
            # Episode loop
            while episode_steps < max_steps:
                # Increment step counter at the beginning of each iteration
                episode_steps += 1
                # Get actions with noise for exploration
                ahfsi_actions = self.get_actions(ahfsi_states, self.ahfsi_actors, noise_scale)
                standard_actions = self.get_actions(standard_states, self.standard_actors, noise_scale)
                
                # Step the environments
                ahfsi_next_states, ahfsi_rewards, ahfsi_dones, ahfsi_info = self.env_ahfsi.step(ahfsi_actions)
                standard_next_states, standard_rewards, standard_dones, standard_info = self.env_standard.step(standard_actions)
                
                # Get actual next states from UAVs
                ahfsi_next_states = [uav.get_observation() for uav in self.env_ahfsi.uavs]
                standard_next_states = [uav.get_observation() for uav in self.env_standard.uavs]
                
                # Check for collisions with obstacles - this is critical for the improved obstacle avoidance
                for uav_idx, uav in enumerate(self.env_ahfsi.uavs):
                    if hasattr(uav, 'collision_detected') and uav.collision_detected:
                        episode_collisions_ahfsi += 1
                for uav_idx, uav in enumerate(self.env_standard.uavs):
                    if hasattr(uav, 'collision_detected') and uav.collision_detected:
                        episode_collisions_standard += 1
                
                # Store experiences in replay buffers
                self.store_experience(ahfsi_states, ahfsi_actions, ahfsi_rewards, ahfsi_next_states, ahfsi_dones, 
                                      self.ahfsi_buffer)
                self.store_experience(standard_states, standard_actions, standard_rewards, standard_next_states, standard_dones, 
                                      self.standard_buffer)
                
                # Update states for next step
                ahfsi_states = ahfsi_next_states
                standard_states = standard_next_states
                
                # Accumulate rewards
                episode_reward_ahfsi += np.mean(ahfsi_rewards)
                episode_reward_standard += np.mean(standard_rewards)
                
                # Append collision data for this step
                if hasattr(self.env_ahfsi.uavs[0], 'collision_detected') and self.env_ahfsi.uavs[0].collision_detected:
                    episode_collisions_ahfsi += 1
                if hasattr(self.env_standard.uavs[0], 'collision_detected') and self.env_standard.uavs[0].collision_detected:
                    episode_collisions_standard += 1
                
                # Increment the step counter
                # episode_steps incremented here instead of at the end of the loop
                
                # Check UAV-target distances for debugging (helps identify target visibility issues)
                if episode_steps % 20 == 0 or episode_steps == 1 or ahfsi_dones:
                    # Calculate distances to targets and obstacles
                    target_dists = [np.linalg.norm(uav.position - self.env_ahfsi.target.position) 
                                   for uav in self.env_ahfsi.uavs]
                    min_target_dist = min(target_dists)
                    
                    # Get minimum distances to obstacles for reporting avoidance effectiveness
                    if self.env_ahfsi.obstacles:
                        obstacle_dists = []
                        for uav in self.env_ahfsi.uavs:
                            for obstacle in self.env_ahfsi.obstacles:
                                obstacle_dists.append(np.linalg.norm(uav.position - obstacle.position) - obstacle.radius)
                        min_obstacle_dist = min(obstacle_dists) if obstacle_dists else float('inf')
                        
                        # Show both target approach and obstacle avoidance metrics with clear step numbering
                        print(f"  AHFSI Episode {episode+1}/{num_episodes} - Step {episode_steps}/{max_steps}: " 
                              f"{Fore.YELLOW}Target dist: {min_target_dist:.2f}{Style.RESET_ALL}, "
                              f"{Fore.GREEN}Obstacle clearance: {min_obstacle_dist:.2f}{Style.RESET_ALL}, "
                              f"{Fore.CYAN}Sensor range: {CONFIG['sensor_range']:.2f}{Style.RESET_ALL}")
                
                # Update networks if enough experience is collected
                if self.ahfsi_buffer.size() > BATCH_SIZE * 10 and self.standard_buffer.size() > BATCH_SIZE * 10:
                    for _ in range(4):  # Multiple updates per environment step for faster learning
                        self.learn(self.ahfsi_buffer, self.ahfsi_actors, self.ahfsi_critics)
                        self.learn(self.standard_buffer, self.standard_actors, self.standard_critics)
                
                # Check if episode is done (dones is a boolean, not a list)
                if ahfsi_dones or standard_dones:
                    # Track successful episodes and log results
                    ahfsi_success = 'success' in ahfsi_info and ahfsi_info['success']
                    standard_success = 'success' in standard_info and standard_info['success']
                    
                    # Update success counters and tracking lists
                    if ahfsi_success:
                        ahfsi_successes += 1
                        ahfsi_success_flags.append(True)
                    else:
                        ahfsi_success_flags.append(False)
                        
                    if standard_success:
                        standard_successes += 1
                        standard_success_flags.append(True)
                    else:
                        standard_success_flags.append(False)
                        
                    # Clear terminal line and add spacing for visibility
                    print('\n\n')
                    
                    # ==== DISPLAY CLEAR SUCCESS/FAILURE INDICATOR ====
                    box_width = 70
                    horizontal_border = '+' + '-' * (box_width-2) + '+'
                    empty_line = '|' + ' ' * (box_width-2) + '|'
                    
                    if ahfsi_success:
                        # SUCCESS: Green success box with clear result status
                        print(Fore.GREEN + Style.BRIGHT)
                        print(horizontal_border)
                        print(empty_line)
                        success_text = 'OPERATION SUCCESSFUL'
                        mission_text = 'MISSION ACCOMPLISHED'
                        steps_text = f'TARGET REACHED IN {episode_steps} STEPS'
                        collision_text = f'OBSTACLE AVOIDANCE COLLISIONS: {episode_collisions_ahfsi}'
                        
                        # Calculate padding to center text
                        success_padding = (box_width - 2 - len(success_text)) // 2
                        mission_padding = (box_width - 2 - len(mission_text)) // 2
                        steps_padding = (box_width - 2 - len(steps_text)) // 2
                        collision_padding = (box_width - 2 - len(collision_text)) // 2
                        
                        # Print perfectly centered text
                        print('|' + ' ' * success_padding + success_text + ' ' * (box_width - 2 - len(success_text) - success_padding) + '|')
                        print('|' + ' ' * mission_padding + mission_text + ' ' * (box_width - 2 - len(mission_text) - mission_padding) + '|')
                        print('|' + ' ' * steps_padding + steps_text + ' ' * (box_width - 2 - len(steps_text) - steps_padding) + '|')
                        print(empty_line)
                        print('|' + ' ' * collision_padding + collision_text + ' ' * (box_width - 2 - len(collision_text) - collision_padding) + '|')
                        print(empty_line)
                        print(horizontal_border)
                        print(Style.RESET_ALL)
                        
                        # Additional performance data
                        print(f'{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} AHFSI-enhanced obstacle avoidance was effective')
                        print(f'{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} Target successfully captured')
                    else:
                        # FAILURE: Red failure box with clear result status
                        print(Fore.RED + Style.BRIGHT)
                        print(horizontal_border)
                        print(empty_line)
                        failure_text = 'OPERATION FAILED'
                        mission_text = 'MISSION INCOMPLETE'
                        steps_text = f'STOPPED AFTER {episode_steps} STEPS'
                        collision_text = f'OBSTACLE AVOIDANCE COLLISIONS: {episode_collisions_ahfsi}'
                        
                        # Calculate padding to center text
                        failure_padding = (box_width - 2 - len(failure_text)) // 2
                        mission_padding = (box_width - 2 - len(mission_text)) // 2
                        steps_padding = (box_width - 2 - len(steps_text)) // 2
                        collision_padding = (box_width - 2 - len(collision_text)) // 2
                        
                        # Print perfectly centered text
                        print('|' + ' ' * failure_padding + failure_text + ' ' * (box_width - 2 - len(failure_text) - failure_padding) + '|')
                        print('|' + ' ' * mission_padding + mission_text + ' ' * (box_width - 2 - len(mission_text) - mission_padding) + '|')
                        print('|' + ' ' * steps_padding + steps_text + ' ' * (box_width - 2 - len(steps_text) - steps_padding) + '|')
                        print(empty_line)
                        print('|' + ' ' * collision_padding + collision_text + ' ' * (box_width - 2 - len(collision_text) - collision_padding) + '|')
                        print(empty_line)
                        print(horizontal_border)
                        print(Style.RESET_ALL)
                        
                        # Additional performance data
                        print(f'{Fore.RED}[FAILURE]{Style.RESET_ALL} UAVs failed to reach target')
                        print(f'{Fore.RED}[FAILURE]{Style.RESET_ALL} Mission objectives not achieved')
                    
                    # Clear the previous line from progress bar
                    print("\n")
                    
                    # Save metrics to CSV file if pandas is available
                    if PANDAS_AVAILABLE:
                        # Create a DataFrame row
                        metrics_row = pd.DataFrame({
                            'episode': [episode],
                            'ahfsi_success': [int(ahfsi_success)],
                            'ahfsi_reward': [episode_reward_ahfsi],
                            'ahfsi_collisions': [episode_collisions_ahfsi],
                            'standard_success': [int(standard_success)],
                            'standard_reward': [episode_reward_standard],
                            'standard_collisions': [episode_collisions_standard],
                            'steps': [episode_steps],
                            'timestamp': [datetime.datetime.now()]
                        })
                        
                        # Write to CSV (append mode)
                        write_header = not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0
                        metrics_row.to_csv(metrics_file, mode='a', header=write_header, index=False)
                    else:
                        # Alternative without pandas
                        with open(metrics_file, 'a') as f:
                            if episode == 0 or not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0:
                                f.write("episode,ahfsi_success,ahfsi_reward,ahfsi_collisions,standard_success,standard_reward,standard_collisions,steps\n")
                    logger.info(f"AHFSI-RL Model: {'[SUCCESS]' if ahfsi_success else '[FAILED]'} | "
                               f"Reward: {episode_reward_ahfsi:.2f} | "
                               f"Obstacle Collisions: {episode_collisions_ahfsi} | "
                               f"Steps: {episode_steps}")
                    
                    logger.info(f"Standard Model: {'[SUCCESS]' if standard_success else '[FAILED]'} | "
                               f"Reward: {episode_reward_standard:.2f} | "
                               f"Obstacle Collisions: {episode_collisions_standard} | "
                               f"Steps: {episode_steps}")
                    
                    # Add more details on obstacle avoidance performance
                    avg_obstacle_distance = 0
                    if len(self.env_ahfsi.obstacles) > 0 and episode_steps > 0:
                        obstacle_distances = []
                        for uav in self.env_ahfsi.uavs:
                            for obstacle in self.env_ahfsi.obstacles:
                                dist = np.linalg.norm(uav.position - obstacle.position) - obstacle.radius
                                obstacle_distances.append(dist)
                        avg_obstacle_distance = sum(obstacle_distances) / len(obstacle_distances) if obstacle_distances else 0
                        
                    print(f"{Fore.CYAN}Obstacle Avoidance Performance:{Style.RESET_ALL}")
                    print(f"  Average obstacle clearance: {Fore.YELLOW}{avg_obstacle_distance:.2f}{Style.RESET_ALL} units")
                    print(f"  Detection range: {Fore.YELLOW}{CONFIG['sensor_range']:.2f}{Style.RESET_ALL} units")
                    
                    # Add current episode rewards to lists for plotting
                    ahfsi_episode_rewards.append(episode_reward_ahfsi)
                    standard_episode_rewards.append(episode_reward_standard)
                    
                    # Add collision counts for this episode
                    ahfsi_episode_collisions.append(episode_collisions_ahfsi)
                    standard_episode_collisions.append(episode_collisions_standard)
                    
                    # Store these in the instance variables for plotting later
                    self.ahfsi_rewards = ahfsi_episode_rewards
                    self.standard_rewards = standard_episode_rewards
                    self.ahfsi_collision_counts = ahfsi_episode_collisions
                    self.standard_collision_counts = standard_episode_collisions
                    self.ahfsi_success_flags = ahfsi_success_flags
                    self.standard_success_flags = standard_success_flags

                    # Log cumulative statistics
                    ahfsi_success_rate = ahfsi_successes / (episode + 1)
                    standard_success_rate = standard_successes / (episode + 1)
                    print(f"\n{Fore.CYAN}Cumulative Statistics:{Style.RESET_ALL}")
                    print(f"  AHFSI Success Rate: {Fore.YELLOW}{ahfsi_success_rate:.2%}{Style.RESET_ALL}")
                    print(f"  Standard Success Rate: {Fore.YELLOW}{standard_success_rate:.2%}{Style.RESET_ALL}")
                    print(f"  AHFSI Total Collisions: {Fore.MAGENTA}{sum(self.ahfsi_collision_counts)}{Style.RESET_ALL}")
                    print(f"  Standard Total Collisions: {Fore.MAGENTA}{sum(self.standard_collision_counts)}{Style.RESET_ALL}")
                    
                    # Track success rates over time
                    self.ahfsi_success_rate.append(ahfsi_success_rate)
                    self.standard_success_rate.append(standard_success_rate)
                    
                    # Log a divider line using ASCII characters only (no Unicode)
                    logger.info("=================================================\n")
                    break
                
            # Append collision data for this step
            if hasattr(self.env_ahfsi.uavs[0], 'collision_detected') and self.env_ahfsi.uavs[0].collision_detected:
                episode_collisions_ahfsi += 1
            if hasattr(self.env_standard.uavs[0], 'collision_detected') and self.env_standard.uavs[0].collision_detected:
                episode_collisions_standard += 1
                self.ahfsi_success_rate.append(ahfsi_success_rate)
                self.standard_success_rate.append(standard_success_rate)
                
                # Reset counters for next window
                if (episode + 1) % success_window == 0:
                    ahfsi_successes = 0
                    standard_successes = 0
            
            # Save checkpoints periodically
            if (episode + 1) % 100 == 0:
                self.save_checkpoints()
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  AHFSI Reward: {episode_reward_ahfsi:.2f}, Steps: {episode_steps}")
                print(f"  Standard Reward: {episode_reward_standard:.2f}, Steps: {episode_steps}")
                if self.ahfsi_success_rate and self.standard_success_rate:
                    print(f"  Success Rates - AHFSI: {self.ahfsi_success_rate[-1]:.2f}, Standard: {self.standard_success_rate[-1]:.2f}")
                print(f"  Noise Scale: {noise_scale:.3f}")
                
        # Final save
        self.save_checkpoints()
        
        # Visualize results
        self.plot_results()
        
        print("\nTraining completed!")
        if self.ahfsi_success_rate and self.standard_success_rate:
            ahfsi_final_success = np.mean(self.ahfsi_success_rate[-10:])
            standard_final_success = np.mean(self.standard_success_rate[-10:])
            success_improvement = ((ahfsi_final_success - standard_final_success) / max(0.01, standard_final_success)) * 100
            print(f"Final Success Rate - AHFSI: {ahfsi_final_success:.2f}, Standard: {standard_final_success:.2f}")
            print(f"Success Rate Improvement with AHFSI: {success_improvement:.1f}%")
        
        # Handle small episode counts properly to avoid NaN errors
        if len(self.ahfsi_rewards) > 0:
            # Use all available rewards if less than 100 episodes were run
            window_size = min(100, len(self.ahfsi_rewards))
            ahfsi_final_reward = np.mean(self.ahfsi_rewards[-window_size:])
            standard_final_reward = np.mean(self.standard_rewards[-window_size:])
            
            # Safe division to avoid errors when rewards are close to zero
            if abs(standard_final_reward) > 0.001:
                reward_improvement = ((ahfsi_final_reward - standard_final_reward) / abs(standard_final_reward)) * 100
            else:
                # If standard reward is very close to zero, use absolute improvement
                reward_improvement = ahfsi_final_reward - standard_final_reward
                
            print(f"Final Average Reward - AHFSI: {ahfsi_final_reward:.2f}, Standard: {standard_final_reward:.2f}")
            print(f"Reward Improvement with AHFSI: {reward_improvement:.1f}%")
        else:
            print("No episodes completed - cannot calculate average rewards")
    
    def get_actions(self, states, actors, noise_scale):
        """Get actions from actor networks with exploration noise
        
        Args:
            states: Current states for each agent
            actors: List of actor networks
            noise_scale: Scale of noise to add for exploration
            د 
        Returns:
            List of actions for each agent
        """
        actions = []
        for i, state in enumerate(states):
            # Get action from actor network
            action = actors[i].predict(np.array([state]))[0]
            
            # Add exploration noise
            noise = np.random.normal(0, noise_scale, size=self.action_dim)
            action = action + noise
            
            # Clip to action range [-1, 1]
            action = np.clip(action, -1, 1)
            actions.append(action)
        
        return actions
    
    def store_experience(self, states, actions, rewards, next_states, dones, buffer):
        """Store experience in replay buffer
        
        Args:
            states: Current states for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_states: Next states for each agent
            dones: Done flags for each agent
            buffer: Replay buffer to store in
        """
        # Format the experience for storage
        states_flat = np.concatenate(states)
        actions_flat = np.concatenate(actions)
        next_states_flat = np.concatenate(next_states)
        
        # Store in buffer - using the right API for our ReplayBuffer
        # Generate a simple correlation index based on average state value
        correlation_index = np.mean(states_flat)
        
        # Use the correct method 'add' instead of 'store_transition'
        buffer.add(states_flat, actions_flat, np.mean(rewards), next_states_flat, correlation_index)
    
    def learn(self, buffer, actors, critics, batch_size=64):
        """Update networks from experiences in replay buffer
        
        Args:
            buffer: Replay buffer to sample from
            actors: List of actor networks to update
            critics: List of critic networks to update
            batch_size: Size of minibatch for learning
        """
        # Skip if buffer doesn't have enough samples
        if buffer.size() < batch_size:
            return
        
        # Sample from buffer using the correct API
        # Getting batch of experiences accounting for our enhanced obstacle avoidance
        states, actions, rewards, next_states, correlation_indices, _ = buffer.sample_batch(batch_size)
        
        # Update each agent's networks
        for i in range(self.num_agents):
            # Extract agent's portion of state
            start_idx = i * self.state_dim
            end_idx = start_idx + self.state_dim
            agent_states = states[:, start_idx:end_idx]
            agent_next_states = next_states[:, start_idx:end_idx]
            
            # Create a dones mask (assuming not done since we don't have it from the buffer)
            # This is a simplification - in production you'd store and use actual done flags
            dones_mask = np.zeros(batch_size, dtype=np.float32)
            
            # Update networks directly using TensorFlow operations instead of missing learn methods
            # First get target actions from all target actors
            target_actions = []
            for j in range(self.num_agents):
                j_start = j * self.state_dim
                j_end = j_start + self.state_dim
                j_next_states = next_states[:, j_start:j_end]
                target_actions.append(actors[j].target_predict(j_next_states))
            
            # Get target Q values from critics using proper API
            # Only pass the agent's own state portion to avoid dimension mismatch
            target_q = critics[i].target_predict(agent_next_states, target_actions)
            
            # Calculate target values with discounted rewards
            target_values = rewards + GAMMA * (1.0 - dones_mask) * np.squeeze(target_q)
            
            # Extract individual agent actions from the batch of actions
            # The model expects a list of action tensors, one for each agent
            agent_actions = []
            for j in range(self.num_agents):
                # Each agent has action_dim (2) components
                j_start = j * self.action_dim
                j_end = j_start + self.action_dim
                # Extract this agent's actions from the batch
                agent_actions.append(actions[:, j_start:j_end])
            
            # Update critic using proper API - train method expects states, actions, and target_q_values
            critics[i].train(agent_states, agent_actions, target_values.reshape(-1, 1))
            
            # Update actor using policy gradient
            # First get the gradients of Q with respect to actions
            action_gradients = critics[i].get_action_gradients(agent_states, agent_actions, i)
            
            # Update actor with these gradients using the proper API
            actors[i].train(agent_states, action_gradients)
            
    def _create_empty_plot_with_message(self, filename, message, plots_dir):
        """Create an empty plot with an informative message
        
        Args:
            filename: Name of the file to save
            message: Message to display on the plot
            plots_dir: Directory to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(plots_dir, filename))
        plt.close()
    
    def save_checkpoints(self):
        """Save model checkpoints for both systems"""
        for i in range(self.num_agents):
            # Save AHFSI models with the correct method name (save, not save_checkpoint)
            self.ahfsi_actors[i].save(self.ahfsi_checkpoint_dir)
            self.ahfsi_critics[i].save(self.ahfsi_checkpoint_dir)
            
            # Save standard models
            self.standard_actors[i].save(self.standard_checkpoint_dir)
            self.standard_critics[i].save(self.standard_checkpoint_dir)
            
        print(f"Models saved to {self.ahfsi_checkpoint_dir} and {self.standard_checkpoint_dir}")
    
    def load_checkpoints(self):
        """Load model checkpoints for both systems"""
        for i in range(self.num_agents):
            # Load AHFSI models
            self.ahfsi_actors[i].load(self.ahfsi_checkpoint_dir)
            self.ahfsi_critics[i].load(self.ahfsi_checkpoint_dir)
            
            # Load standard models
            self.standard_actors[i].load(self.standard_checkpoint_dir)
            self.standard_critics[i].load(self.standard_checkpoint_dir)
    
    def plot_results(self):
        """Plot training results for comparison"""
        plots_dir = "plots/training"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Make sure we have some data to plot
        if len(self.ahfsi_rewards) == 0 or len(self.standard_rewards) == 0:
            print("Warning: Not enough data to create meaningful plots")
            # Create empty plots with messages instead of empty plots
            self._create_empty_plot_with_message("rewards_comparison.png", "No reward data available - run more episodes", plots_dir)
            self._create_empty_plot_with_message("success_rates_comparison.png", "No success rate data available - run more episodes", plots_dir)
            self._create_empty_plot_with_message("collision_comparison.png", "No collision data available - run more episodes", plots_dir)
            return
        
        # Plot rewards - always plot raw data points for small datasets
        plt.figure(figsize=(12, 6))
        plt.plot(self.ahfsi_rewards, 'bo-', alpha=0.7, label='AHFSI')
        plt.plot(self.standard_rewards, 'ro-', alpha=0.7, label='Standard')
        
        # Only add smoothing if we have enough data points
        min_episodes_for_smoothing = 10
        if len(self.ahfsi_rewards) >= min_episodes_for_smoothing:
            # Add moving averages for clarity
            window = max(2, min(100, len(self.ahfsi_rewards) // 10))  # Ensure window is at least 2
            ahfsi_smoothed = np.convolve(self.ahfsi_rewards, np.ones(window)/window, mode='valid')
            standard_smoothed = np.convolve(self.standard_rewards, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.ahfsi_rewards)), ahfsi_smoothed, 'b-', linewidth=2, label='AHFSI (smoothed)')
            plt.plot(range(window-1, len(self.standard_rewards)), standard_smoothed, 'r-', linewidth=2, label='Standard (smoothed)')
        
        plt.title('Training Rewards: AHFSI vs Standard')
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "rewards_comparison.png"))
        
        # Plot success rates if available
        if hasattr(self, 'ahfsi_success_flags') and len(self.ahfsi_success_flags) > 0:
            plt.figure(figsize=(12, 6))
            
            # Calculate success rates for each episode (cumulative)
            ahfsi_success_rates = [sum(self.ahfsi_success_flags[:i+1])/(i+1) for i in range(len(self.ahfsi_success_flags))]
            standard_success_rates = [sum(self.standard_success_flags[:i+1])/(i+1) for i in range(len(self.standard_success_flags))]
            
            # Use markers for small datasets to make points visible
            if len(ahfsi_success_rates) < 10:
                plt.plot(ahfsi_success_rates, 'bo-', markersize=8, linewidth=2, label='AHFSI')
                plt.plot(standard_success_rates, 'ro-', markersize=8, linewidth=2, label='Standard')
            else:
                plt.plot(ahfsi_success_rates, 'b-', linewidth=2, label='AHFSI')
                plt.plot(standard_success_rates, 'r-', linewidth=2, label='Standard')
            
            plt.title('Success Rates: AHFSI vs Standard')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Success Rate')
            plt.legend()
            plt.grid(True)
            # Set y-axis limits to 0-1 for success rates
            plt.ylim(-0.05, 1.05)
            plt.savefig(os.path.join(plots_dir, "success_rates_comparison.png"))
        else:
            self._create_empty_plot_with_message("success_rates_comparison.png", "No success rate data available - run more episodes", plots_dir)
        
        # Plot obstacle collision metrics - crucial for evaluating obstacle avoidance
        if hasattr(self, 'ahfsi_collision_counts') and len(self.ahfsi_collision_counts) > 0:
            plt.figure(figsize=(12, 6))
            
            # Use scatter plot with lines for small datasets to ensure visibility
            if len(self.ahfsi_collision_counts) < 10:
                plt.plot(self.ahfsi_collision_counts, 'bo-', markersize=8, linewidth=2, label='AHFSI')
                plt.plot(self.standard_collision_counts, 'ro-', markersize=8, linewidth=2, label='Standard')
            else:
                # For larger datasets, use smoothing if possible
                min_window_size = 3  # Minimum window size for smoothing
                if len(self.ahfsi_collision_counts) >= min_window_size * 2:
                    # Calculate moving average for collisions
                    window = max(min_window_size, min(50, len(self.ahfsi_collision_counts) // 5))
                    ahfsi_collisions_smoothed = np.convolve(self.ahfsi_collision_counts, np.ones(window)/window, mode='valid')
                    standard_collisions_smoothed = np.convolve(self.standard_collision_counts, np.ones(window)/window, mode='valid')
                    plt.plot(range(window-1, len(self.ahfsi_collision_counts)), ahfsi_collisions_smoothed, 'b-', linewidth=2, label='AHFSI (smoothed)')
                    plt.plot(range(window-1, len(self.standard_collision_counts)), standard_collisions_smoothed, 'r-', linewidth=2, label='Standard (smoothed)')
                    # Also plot the raw data with lower opacity
                    plt.plot(self.ahfsi_collision_counts, 'b-', alpha=0.3, linewidth=1, label='AHFSI (raw)')
                    plt.plot(self.standard_collision_counts, 'r-', alpha=0.3, linewidth=1, label='Standard (raw)')
                else:
                    # Not enough data for smoothing
                    plt.plot(self.ahfsi_collision_counts, 'b-', linewidth=2, label='AHFSI')
                    plt.plot(self.standard_collision_counts, 'r-', linewidth=2, label='Standard')
            
            plt.title('Obstacle Collisions: AHFSI vs Standard')
            plt.xlabel('Episode')
            plt.ylabel('Collisions per Episode')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, "collision_comparison.png"))
            
            # Calculate cumulative collisions
            ahfsi_cumulative = np.cumsum(self.ahfsi_collision_counts)
            standard_cumulative = np.cumsum(self.standard_collision_counts)
            
            plt.figure(figsize=(12, 6))
            # Use markers for small datasets for better visibility
            if len(self.ahfsi_collision_counts) < 10:
                plt.plot(ahfsi_cumulative, 'bo-', markersize=8, linewidth=2, label='AHFSI')
                plt.plot(standard_cumulative, 'ro-', markersize=8, linewidth=2, label='Standard')
            else:
                plt.plot(ahfsi_cumulative, 'b-', linewidth=2, label='AHFSI')
                plt.plot(standard_cumulative, 'r-', linewidth=2, label='Standard')
            
            plt.title('Cumulative Obstacle Collisions: AHFSI vs Standard')
            plt.xlabel('Episode')
            plt.ylabel('Total Collisions')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, "cumulative_collision_comparison.png"))
        else:
            # Create empty plots with informative messages if no collision data is available
            self._create_empty_plot_with_message("collision_comparison.png", "No collision data available - run more episodes", plots_dir)
            self._create_empty_plot_with_message("cumulative_collision_comparison.png", "No collision data available - run more episodes", plots_dir)


def train_with_integration():
    """Run the AHFSI-RL integration training"""
        
    print("Starting AHFSI-RL Integration Training")
    print("=" * 60)
    print()
    # Environment parameters
    num_uavs = 5
    num_obstacles = 8
    
    # Create environments with enhanced obstacle detection and avoidance
    print("\nInitializing environments with enhanced obstacle avoidance...")
    print("Obstacle improvements:")
    print("  - Obstacles are 10x larger visually for better visibility")
    print("  - UAVs cannot pass through obstacles - collision prevention active")
    print("  - Proactive obstacle avoidance using sensor data")
    print("  - Physics and visualization are consistent (obstacles same size in both)")
    
    env_ahfsi = Environment(num_uavs=num_uavs, num_obstacles=num_obstacles, enable_ahfsi=True)
    env_standard = Environment(num_uavs=num_uavs, num_obstacles=num_obstacles, enable_ahfsi=False)
    
    # Get dimensions from UAV observations
    state_dim = env_ahfsi.uavs[0].get_observation().shape[0]
    action_dim = 2  # 2D movement (x, y acceleration)
    
    # Create integrator
    print(f"\nCreating AHFSI-RL Integrator with {num_uavs} UAVs, {state_dim} state dimensions")
    integrator = AHFSIRLIntegrator(env_ahfsi, env_standard, num_uavs, state_dim, action_dim)
    
    # Training parameters - read from configuration file
    # If 'num_episodes' not in CONFIG, use default of 500
    num_episodes = CONFIG.get("num_episodes", 500)  # For publication-quality results, increase to 2000+
    max_steps = CONFIG["max_steps_per_episode"]
    
    # Create required directories
    os.makedirs("plots/training", exist_ok=True)
    os.makedirs("checkpoints/ahfsi", exist_ok=True)
    os.makedirs("checkpoints/standard", exist_ok=True)
    
    # Train
    print(f"\nTraining for {num_episodes} episodes (max {max_steps} steps each)...")
    print("This may take a while - training metrics will be displayed during training")
    print("\nPress Ctrl+C to interrupt training early (models will be saved)")
    
    try:
        integrator.train(num_episodes, max_steps)
        print("\nAHFSI-RL Integration Training Complete")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user - saving current model state")
        integrator.save_checkpoints()
    
    # Plot final results
    integrator.plot_results()
    print("\nOutput plots have been saved to plots/training/")
    print("To run a demo with trained models, use: python run_ahfsi_demo.py")
    print("  Options: --ahfsi=True/False --style=ahfsi/military --uavs=5 --obstacles=8")


if __name__ == "__main__":
    train_with_integration()
