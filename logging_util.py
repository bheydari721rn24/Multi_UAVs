import os
import time
import numpy as np
from datetime import datetime

class SimulationLogger:
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate a unique log file name based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
        
        # Initialize metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'success_rate': [],
            'episode_steps': [],
            'capture_times': [],
            'random_seeds': [],
            'initial_positions': {}
        }
        
        # Open log file and write header
        with open(self.log_file, 'w') as f:
            f.write(f"Multi-UAV Simulation Log - Started at {timestamp}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_config(self, config):
        """Log configuration parameters"""
        with open(self.log_file, 'a') as f:
            f.write("Configuration Parameters:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    def log_episode_start(self, episode, seed, positions):
        """Log the start of an episode with initial positions"""
        self.metrics['random_seeds'].append(seed)
        self.metrics['initial_positions'][episode] = positions
        
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode} - Started (Seed: {seed})\n")
            f.write("  Initial Positions:\n")
            for entity_type, pos_list in positions.items():
                f.write(f"    {entity_type}: {pos_list}\n")
    
    def log_episode_end(self, episode, reward, success, steps):
        """Log the end of an episode with results"""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['success_rate'].append(1 if success else 0)
        self.metrics['episode_steps'].append(steps)
        
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode} - Completed\n")
            f.write(f"  Reward: {reward:.4f}\n")
            f.write(f"  Success: {success}\n")
            f.write(f"  Steps: {steps}\n")
            f.write("  " + "-" * 60 + "\n\n")
    
    def log_step(self, episode, step, actions, state, rewards):
        """Log details of a specific step (optional, can generate large logs)"""
        # This could generate very large logs, so use sparingly
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode}, Step {step}:\n")
            f.write(f"  Actions: {actions}\n")
            f.write(f"  Rewards: {rewards}\n")
            # f.write(f"  State: {state[:10]}...\n")  # Print start of state vector
    
    def log_capture(self, episode, step):
        """Log when target is captured"""
        self.metrics['capture_times'].append(step)
        
        with open(self.log_file, 'a') as f:
            f.write(f"Episode {episode} - Target captured at step {step}\n")
    
    def log_error(self, message):
        """Log error messages"""
        with open(self.log_file, 'a') as f:
            f.write(f"ERROR: {message}\n")
    
    def get_summary_stats(self):
        """Return summary statistics of the simulation"""
        if not self.metrics['episode_rewards']:
            return "No episodes completed yet."
        
        avg_reward = np.mean(self.metrics['episode_rewards'])
        success_rate = np.mean(self.metrics['success_rate']) if self.metrics['success_rate'] else 0
        avg_steps = np.mean(self.metrics['episode_steps']) if self.metrics['episode_steps'] else 0
        avg_capture_time = np.mean(self.metrics['capture_times']) if self.metrics['capture_times'] else 0
        
        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_capture_time': avg_capture_time,
            'total_episodes': len(self.metrics['episode_rewards'])
        }
    
    def save_summary(self):
        """Save summary statistics to log file"""
        stats = self.get_summary_stats()
        if isinstance(stats, str):
            return
        
        with open(self.log_file, 'a') as f:
            f.write("\nSimulation Summary:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Episodes: {stats['total_episodes']}\n")
            f.write(f"Average Reward: {stats['avg_reward']:.4f}\n")
            f.write(f"Success Rate: {stats['success_rate']:.4f}\n")
            f.write(f"Average Steps per Episode: {stats['avg_steps']:.2f}\n")
            f.write(f"Average Capture Time: {stats['avg_capture_time']:.2f}\n")
            f.write("=" * 80 + "\n")
