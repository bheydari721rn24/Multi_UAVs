import os
import time
import numpy as np
from datetime import datetime
import logging
import colorama
from colorama import Fore, Back, Style

# Initialize colorama with autoreset and force color mode
colorama.init(autoreset=True, convert=True, strip=False, wrap=True)

# Define log levels with their colors
class LogLevel:
    DEBUG = 'DEBUG'
    INFO = 'INFO'
    SUCCESS = 'SUCCESS'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
    PROGRESS = 'PROGRESS'  # New level for progress information

# Define color mappings for terminal output
LOG_COLORS = {
    LogLevel.DEBUG: Fore.CYAN,
    LogLevel.INFO: Fore.WHITE,
    LogLevel.SUCCESS: Fore.GREEN,
    LogLevel.WARNING: Fore.YELLOW,
    LogLevel.ERROR: Fore.RED,
    LogLevel.CRITICAL: Fore.RED + Style.BRIGHT,
    LogLevel.PROGRESS: Fore.MAGENTA + Style.BRIGHT  # Bright magenta (purple) for progress
}

# Define background colors for special messages
PROGRESS_BG = Back.MAGENTA
SUCCESS_BG = Back.GREEN
ERROR_BG = Back.RED

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to terminal logs"""
    
    def format(self, record):
        # Save original levelname
        levelname = record.levelname
        
        # Apply color based on level
        if levelname in LOG_COLORS:
            record.levelname = f"{LOG_COLORS[levelname]}{levelname}{Style.RESET_ALL}"
        
        # Special background highlighting for important messages
        if hasattr(record, 'highlight') and record.highlight:
            if levelname == LogLevel.SUCCESS or levelname == 'SUCCESS':
                record.msg = f"{SUCCESS_BG}{Fore.WHITE}{record.msg}{Style.RESET_ALL}"
            elif levelname == LogLevel.ERROR or levelname == 'ERROR' or levelname == LogLevel.CRITICAL or levelname == 'CRITICAL':
                record.msg = f"{ERROR_BG}{Fore.WHITE}{record.msg}{Style.RESET_ALL}"
            elif levelname == LogLevel.PROGRESS or levelname == 'PROGRESS':
                record.msg = f"{PROGRESS_BG}{Fore.WHITE}{record.msg}{Style.RESET_ALL}"
            
        # If it's a SUCCESS message (custom level)
        elif hasattr(record, 'success') and record.success:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = levelname
        
        return result

class SimulationLogger:
    def __init__(self, log_dir='./logs', console_level=LogLevel.INFO, file_level=LogLevel.DEBUG):
        """
        Initialize a new SimulationLogger.
        
        Args:
            log_dir (str): Directory where log files will be stored
            console_level (str): Minimum level to display in console
            file_level (str): Minimum level to write to log file
        """
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
            'initial_positions': {},
            'start_time': datetime.now()
        }
        
        # Configure logger
        self.logger = logging.getLogger('simulation')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create handlers with proper encoding for Windows terminals
        import sys
        console_handler = logging.StreamHandler(sys.stdout)
        # Use UTF-8 encoding for the log file
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        
        # Set levels
        console_handler.setLevel(self._get_level(console_level))
        file_handler.setLevel(self._get_level(file_level))
        
        # Create formatters
        console_format = '%(asctime)s - %(levelname)s - %(message)s'
        file_format = '%(asctime)s - %(levelname)s - %(message)s'
        console_formatter = ColoredFormatter(console_format, datefmt='%H:%M:%S')
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Set formatters
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

        # Force immediate output to terminal (no buffering)
        console_handler.flush()
        
        # Add SUCCESS level (between INFO and WARNING)
        logging.SUCCESS = 25  # Between INFO (20) and WARNING (30)
        logging.addLevelName(logging.SUCCESS, LogLevel.SUCCESS)
        
        # Add PROGRESS level (between INFO and SUCCESS)
        logging.PROGRESS = 22  # Between INFO (20) and SUCCESS (25)
        logging.addLevelName(logging.PROGRESS, LogLevel.PROGRESS)

        # Add method to logger
        def success(self, message, *args, **kwargs):
            highlight = kwargs.pop('highlight', False)
            extra = kwargs.pop('extra', {})
            extra['highlight'] = highlight
            self.log(logging.SUCCESS, message, *args, extra=extra, **kwargs)
        
        # Add progress method
        def progress(self, message, *args, **kwargs):
            highlight = kwargs.pop('highlight', False)
            extra = kwargs.pop('extra', {})
            extra['highlight'] = highlight
            self.log(logging.PROGRESS, message, *args, extra=extra, **kwargs)
            # Force immediate output to terminal
            for handler in self.handlers:
                handler.flush()
        
        logging.Logger.success = success
        logging.Logger.progress = progress
        
        # Print header with clear separation and colors
        header = f"\n{Back.BLUE}{Fore.WHITE} MULTI-UAV SIMULATION STARTED AT {timestamp} {Style.RESET_ALL}\n"
        print(header)
        
        # Log header info
        self.logger.info(f"Multi-UAV Simulation Log - Started at {timestamp}")
        self.logger.info("=" * 80)
    
    def _get_level(self, level_name):
        """Convert level name to logging level integer"""
        if level_name == LogLevel.DEBUG:
            return logging.DEBUG
        elif level_name == LogLevel.INFO:
            return logging.INFO
        elif level_name == LogLevel.SUCCESS:
            return logging.SUCCESS
        elif level_name == LogLevel.PROGRESS:
            return logging.PROGRESS
        elif level_name == LogLevel.WARNING:
            return logging.WARNING
        elif level_name == LogLevel.ERROR:
            return logging.ERROR
        elif level_name == LogLevel.CRITICAL:
            return logging.CRITICAL
        else:
            return logging.INFO
    
    def log_config(self, config):
        """Log configuration parameters"""
        self.logger.info("Configuration Parameters:")
        
        # Extract key parameters for console display
        key_params = {
            'mode': config.get('mode', 'unknown'),
            'num_uavs': config.get('num_uavs', 0),
            'num_obstacles': config.get('num_obstacles', 0),
            'episodes': config.get('episodes', 0),
            'dynamic_obstacles': config.get('dynamic_obstacles', False),
        }
        
        # Print key configuration parameters with highlighting
        self.logger.progress(
            f"Training with {key_params['num_uavs']} UAVs, {key_params['num_obstacles']} obstacles "
            f"for {key_params['episodes']} episodes. "
            f"Dynamic obstacles: {'ON' if key_params['dynamic_obstacles'] else 'OFF'}",
            highlight=True
        )
        
        # Log all parameters to file
        for key, value in config.items():
            self.logger.debug(f"  {key}: {value}")
    
    def log_episode_start(self, episode, seed, positions, total_episodes=None):
        """Log the start of an episode with initial positions"""
        self.metrics['random_seeds'].append(seed)
        self.metrics['initial_positions'][episode] = positions
        
        # Calculate progress information if total_episodes is known
        if total_episodes:
            # Calculate elapsed time and estimated time remaining
            elapsed = datetime.now() - self.metrics['start_time']
            if episode > 0:  # Avoid division by zero
                time_per_episode = elapsed / episode
                remaining_episodes = total_episodes - episode
                est_time_remaining = time_per_episode * remaining_episodes
                
                # Format time strings
                elapsed_str = self._format_time(elapsed.total_seconds())
                remaining_str = self._format_time(est_time_remaining.total_seconds())
                
                # Log progress with completion percentage
                progress_pct = (episode / total_episodes) * 100
                progress_bar = self._create_progress_bar(progress_pct)
                
                self.logger.progress(
                    f"Progress: {progress_bar} {progress_pct:.1f}% | "
                    f"Episode {episode}/{total_episodes} | "
                    f"Elapsed: {elapsed_str} | Remaining: {remaining_str}",
                    highlight=True
                )
            else:
                # First episode, just show basic info
                self.logger.progress(f"Starting Episode {episode}/{total_episodes}", highlight=True)
        else:
            # If total_episodes is not known, just log the episode start
            self.logger.info(f"Episode {episode} - Started (Seed: {seed})")
        
        # Log positions at debug level (detailed)
        for entity_type, pos_list in positions.items():
            self.logger.debug(f"  Initial {entity_type}: {pos_list}")
    
    def _format_time(self, seconds):
        """Format seconds into a human-readable string (HH:MM:SS)"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _create_progress_bar(self, percentage, width=20):
        """Create a simple progress bar for display"""
        filled_width = int(width * percentage / 100)
        bar = '█' * filled_width + '░' * (width - filled_width)
        return f"[{bar}]"
    
    def log_episode_end(self, episode, reward, success, steps, total_episodes=None):
        """Log the end of an episode with results"""
        self.metrics['episode_rewards'].append(reward)
        self.metrics['success_rate'].append(1 if success else 0)
        self.metrics['episode_steps'].append(steps)
        
        # Calculate recent success rate for last 20 episodes (or fewer if not available)
        recent_window = min(20, len(self.metrics['success_rate']))
        if recent_window > 0:
            recent_success_rate = sum(self.metrics['success_rate'][-recent_window:]) / recent_window
        else:
            recent_success_rate = 0
        
        # Add success status to the log message
        status = "SUCCESS" if success else "FAILED"
        if success:
            # Highlight successful capture
            self.logger.success(
                f"Episode {episode} - {status} in {steps} steps | Reward: {reward:.4f}",
                highlight=episode % 10 == 0  # Highlight every 10th episode
            )
        else:
            self.logger.info(f"Episode {episode} - {status} in {steps} steps | Reward: {reward:.4f}")
        
        # Every 10 episodes, show a summary of recent performance
        if episode % 10 == 0 and episode > 0:
            self.logger.progress(
                f"Last {recent_window} episodes: "
                f"Success rate {recent_success_rate:.2f} | "
                f"Avg reward {np.mean(self.metrics['episode_rewards'][-recent_window:]):.2f} | "
                f"Avg steps {np.mean(self.metrics['episode_steps'][-recent_window:]):.1f}",
                highlight=True
            )
    
    def log_step(self, episode, step, actions, state, rewards):
        """Log details of a specific step (optional, can generate large logs)"""
        # Use INFO level to ensure steps are visible in console
        self.logger.info(f"Episode {episode}, Step {step}:")
        # More detailed info at debug level
        self.logger.debug(f"  Actions: {actions}")
        self.logger.debug(f"  Rewards: {rewards}")
        # self.logger.debug(f"  State: {state[:10]}...")  # Print start of state vector
    
    def log_capture(self, episode, step):
        """Log when target is captured"""
        self.metrics['capture_times'].append(step)
        self.logger.success(f"Episode {episode} - Target captured at step {step}!", highlight=True)
    
    def log_error(self, message):
        """Log error messages"""
        self.logger.error(message)
    
    def log_warning(self, message):
        """Log warning messages"""
        self.logger.warning(message)
    
    def log_debug(self, message):
        """Log debug information"""
        self.logger.debug(message)
    
    def log_progress(self, message, highlight=False):
        """Log progress information"""
        self.logger.progress(message, highlight=highlight)
        
    def log_critical(self, message):
        """Log critical errors"""
        self.logger.critical(message)
    
    def get_summary_stats(self):
        """Return summary statistics of the simulation"""
        if not self.metrics['episode_rewards']:
            return "No episodes completed yet."
        
        avg_reward = np.mean(self.metrics['episode_rewards'])
        success_rate = np.mean(self.metrics['success_rate']) if self.metrics['success_rate'] else 0
        avg_steps = np.mean(self.metrics['episode_steps']) if self.metrics['episode_steps'] else 0
        avg_capture_time = np.mean(self.metrics['capture_times']) if self.metrics['capture_times'] else 0
        
        # Calculate elapsed time
        elapsed_time = datetime.now() - self.metrics['start_time']
        elapsed_str = self._format_time(elapsed_time.total_seconds())
        
        return {
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'avg_capture_time': avg_capture_time,
            'total_episodes': len(self.metrics['episode_rewards']),
            'elapsed_time': elapsed_str
        }
    
    def save_summary(self):
        """Save summary statistics to log file"""
        stats = self.get_summary_stats()
        if isinstance(stats, str):
            return
        
        summary_text = (
            "\nSimulation Summary:\n"
            f"{'=' * 80}\n"
            f"Total Episodes: {stats['total_episodes']}\n"
            f"Average Reward: {stats['avg_reward']:.4f}\n"
            f"Success Rate: {stats['success_rate']:.4f}\n"
            f"Average Steps per Episode: {stats['avg_steps']:.2f}\n"
            f"Average Capture Time: {stats['avg_capture_time']:.2f}\n"
            f"Total Elapsed Time: {stats['elapsed_time']}\n"
            f"{'=' * 80}\n"
        )
        
        # Print to console with colors
        print(f"\n{Back.BLUE}{Fore.WHITE} SIMULATION SUMMARY {Style.RESET_ALL}")
        print(f"{Fore.CYAN}{summary_text}{Style.RESET_ALL}")
        
        # Log to file
        self.logger.info(summary_text)
        
    def close(self):
        """Close the logger and handlers"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)
        print(f"\n{Back.GREEN}{Fore.WHITE} LOGGING COMPLETED {Style.RESET_ALL}\n")
