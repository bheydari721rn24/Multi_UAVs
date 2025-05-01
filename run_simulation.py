import numpy as np
import tensorflow as tf
import time
import argparse
import os
import matplotlib.pyplot as plt
import random

# Set TensorFlow logging level (reduce noise in console)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR
from datetime import datetime
from environment import Environment
from cel_maddpg import CEL_MADDPG
from visualization import Visualizer
from utils import CONFIG, create_directory

# Try to import the logger, if available
try:
    from logging_util import SimulationLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    print("Warning: logging_util module not found. Running without detailed logging.")

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-UAV Roundup Strategy with CEL-MADDPG')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo'],
                        help='Mode: train, test, or demo')
    parser.add_argument('--num_uavs', type=int, default=3, help='Number of UAVs')
    parser.add_argument('--num_obstacles', type=int, default=3, help='Number of obstacles')
    parser.add_argument('--dynamic_obstacles', action='store_true', help='Enable dynamic obstacles')
    parser.add_argument('--episodes', type=int, default=CONFIG["num_episodes"], help='Number of training episodes')
    parser.add_argument('--load_path', type=str, default='./saved_models', help='Path to load models from')
    parser.add_argument('--save_path', type=str, default='./saved_models', help='Path to save models to')
    parser.add_argument('--output', type=str, default='./Output-f', help='Output animation filename')
    parser.add_argument('--render_interval', type=int, default=100, 
                        help='Render every N episodes during training')
    parser.add_argument('--save_interval', type=int, default=1000, 
                        help='Save models every N episodes during training')
    parser.add_argument('--seed', type=int, default=None, 
                        help='Random seed. If None, will use time-based random seed')
    parser.add_argument('--logging', action='store_true', default=True,
                        help='Enable detailed logging of simulation data')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of demo runs for visualization with different random seeds')
    return parser.parse_args()

def train(args):
    print("Starting training...")
    
    # Create save directory
    create_directory(args.save_path)
    
    # Create environment
    env = Environment(
        num_uavs=args.num_uavs,
        num_obstacles=args.num_obstacles,
        dynamic_obstacles=args.dynamic_obstacles
    )
    
    # Initialize the CEL-MADDPG algorithm
    state_dim = env.get_state_dim()
    action_dim = 2  # 2D forces [Fx, Fy]
    
    cel_maddpg = CEL_MADDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=args.num_uavs,
        save_path=args.save_path,
        load_path=args.load_path
    )
    
    # Initialize visualization
    visualizer = Visualizer(env)
    
    # For tracking progress
    rewards_history = []
    success_counts = []
    success_window = [0] * 1000  # Track success in last 1000 episodes
    
    # Debug: print initial state of the episode
    print(f"\nInitial UAV positions: {[uav.position.tolist() for uav in env.uavs]}")
    print(f"Initial Target position: {env.target.position.tolist()}")
    print(f"Obstacles: {len(env.obstacles)} (first obstacle at {env.obstacles[0].position.tolist() if env.obstacles else 'none'})")

    
    for episode in range(args.episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        
        # Episode loop
        for step in range(CONFIG["max_steps_per_episode"]):
            # Get actions from policy
            actions = cel_maddpg.get_actions(state)
            
            # Take action in environment
            next_state, rewards, done, info = env.step(actions)
            
            # Calculate correlation index for current state
            corr_index = cel_maddpg.calculate_correlation_index(env)
            
            # Store experience in replay buffer
            cel_maddpg.store_experience(state, actions, rewards, next_state, corr_index)
            
            # Train if enough samples are available
            # Debug print to track buffer size
            if episode < 5 or episode % 10 == 0:  # Print only for first 5 episodes then every 10
                print(f"\nBuffer size: {cel_maddpg.replay_buffer.size()}, Required: {CONFIG['pre_batch_size']}")
                
            if cel_maddpg.replay_buffer.size() > CONFIG["pre_batch_size"]:
                print(f"Training at step {step} in episode {episode}")
                cel_maddpg.train(corr_index)
                # Debug: Check if weights are changing
                if step == 0 and (episode < 5 or episode % 100 == 0):
                    for i, actor in enumerate(cel_maddpg.actors):
                        if actor.model.layers and actor.model.layers[0].weights:
                            print(f"Actor {i} weights sample: {actor.model.layers[0].weights[0][0][0:3]}")
            
            # Render if needed
            if episode % args.render_interval == 0:
                visualizer.render(episode, step)
                time.sleep(0.05)
            
            # Update state and reward
            state = next_state
            episode_reward += sum(rewards)
            
            if done:
                # Update success tracking
                success = 1 if info["success"] else 0
                success_window[episode % 1000] = success
                break
        
        # Episode complete
        rewards_history.append(episode_reward)
        success_rate = sum(success_window) / min(episode + 1, 1000)
        success_counts.append(success_rate)
        
        # Save models periodically
        if episode % args.save_interval == 0 and episode > 0:
            cel_maddpg.save_models(suffix=f"_ep{episode}")
            # Also save a plot of training progress
            if len(rewards_history) > 10:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(rewards_history)
                plt.title('Episode Rewards')
                plt.xlabel('Episode')
                plt.ylabel('Total Reward')
                
                plt.subplot(1, 2, 2)
                plt.plot(success_counts)
                plt.title('Success Rate (1000 episode window)')
                plt.xlabel('Episode')
                plt.ylabel('Success Rate')
                
                plt.savefig(f"{args.save_path}/training_progress_ep{episode}.png")
                plt.close()
        
        # Print progress with more debugging info
        task_state = info.get('task_state', 'unknown') if 'info' in locals() else 'unknown'
        print(f"Episode {episode}/{args.episodes} - Reward: {episode_reward:.2f}, Task: {task_state}, Success Rate: {success_rate:.4f}")
    
    # Save final model
    cel_maddpg.save_models(suffix="_final")
    print("Training completed!")
    
    # Save final animation
    if len(visualizer.stored_frames) > 0:
        visualizer.save_animation(f"{args.save_path}/training_animation.mp4")
    
    # Close visualization
    visualizer.close()

def test(args):
    print("Starting testing...")
    
    # Create environment
    env = Environment(
        num_uavs=args.num_uavs,
        num_obstacles=args.num_obstacles,
        dynamic_obstacles=args.dynamic_obstacles
    )
    
    # Initialize the CEL-MADDPG algorithm
    state_dim = env.get_state_dim()
    action_dim = 2  # 2D forces [Fx, Fy]
    
    cel_maddpg = CEL_MADDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        num_agents=args.num_uavs,
        load_path=args.load_path or args.save_path
    )
    
    # Initialize visualization
    visualizer = Visualizer(env)
    
    # Testing loop
    success_count = 0
    
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0
        
        if not os.path.exists(f"{args.output}\\{episode}"):
            os.makedirs(f"{args.output}\\{episode}")

        for step in range(CONFIG["max_steps_per_episode"]):
            # Get action without exploration noise
            actions = cel_maddpg.get_actions(state, add_noise=False)
            
            # Take action in environment
            next_state, rewards, done, info = env.step(actions)
            
            # Render
            visualizer.render(episode, step)
            time.sleep(0.1)  # Slow down for better visualization
            
            # Update state and reward
            state = next_state
            episode_reward += sum(rewards)
            
            if done:
                if info["success"]:
                    success_count += 1
                break
        
            print(f"Test Episode {episode + 1}/{args.episodes} - "
                  f"Reward: {episode_reward:.2f}, Success: {info['success']}")
        
            # Save final frame
            visualizer.save_image(f"{args.output}\\{episode}\\test_episode_{step}_final.png")
    
        # Save animation
        visualizer.save_animation(f"{args.output}\\{episode}\\test_animation.mp4")
    
    success_rate = success_count / args.episodes
    print(f"Testing completed! Success rate: {success_rate:.4f}")
    
    # Close visualization
    visualizer.close()

def demo(args):
    """Run a live demo with real-time visualization and save animation."""
    print("Starting demo mode...")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output}_{timestamp}" if args.num_runs > 1 else args.output
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Initialize logging if enabled
    if args.logging and LOGGER_AVAILABLE:
        log_dir = os.path.join(output_dir, 'logs')
        main_logger = SimulationLogger(log_dir=log_dir)
        main_logger.log_config({
            "num_uavs": args.num_uavs,
            "num_obstacles": args.num_obstacles,
            "dynamic_obstacles": args.dynamic_obstacles,
            "num_runs": args.num_runs,
            "base_seed": args.seed
        })
    
    # Run multiple demos if requested
    for run_idx in range(args.num_runs):
        print(f"\nStarting run {run_idx+1}/{args.num_runs}")
        
        # Generate a seed for this run
        if args.seed is not None:
            # If seed is provided, create deterministic but different seeds for each run
            current_seed = args.seed + run_idx
        else:
            # Otherwise use time-based random seeds
            current_seed = int(time.time() * 1000) % 100000 + run_idx
        
        print(f"Using seed: {current_seed}")
        
        # Create environment with the current seed
        env = Environment(
            num_uavs=args.num_uavs,
            num_obstacles=args.num_obstacles,
            dynamic_obstacles=args.dynamic_obstacles,
            seed=current_seed,
            enable_logging=args.logging
        )
        
        # Initialize the CEL-MADDPG algorithm
        state_dim = env.get_state_dim()
        action_dim = 2  # 2D forces [Fx, Fy]
        
        agent = CEL_MADDPG(
            state_dim=state_dim,
            action_dim=action_dim,
            num_agents=args.num_uavs,
            load_path=args.load_path or args.save_path
        )
        
        # Initialize visualizer
        visualizer = Visualizer(env)
        
        # Set maximum steps for this run
        max_steps = CONFIG["max_steps_per_episode"]
        
        # Create run-specific subdirectory
        run_dir = os.path.join(output_dir, f"run_{run_idx+1}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # Run simulation with current seed
        state = env.reset(specific_seed=current_seed)  # Use specific seed for reproducibility
        episode_reward = 0
        success = False
        
        for step in range(max_steps):
            # Get actions and update environment
            actions = agent.get_actions(state, add_noise=False)  # No exploration noise in demo
            next_state, rewards, done, info = env.step(actions)
            
            # Render and capture frame
            visualizer.render(run_idx, step)
            
            # Save intermediate frames at regular intervals
            if step % 10 == 0 or done:
                visualizer.save_image(f"{run_dir}/step_{step:03d}.png")
            
            # For real-time viewing, add a small delay
            time.sleep(0.05)
            
            # Update state and reward
            state = next_state
            episode_reward += sum(rewards)
            
            if done:
                success = info['success']
                print(f"Run {run_idx+1} completed at step {step}. Success: {success}")
                # Save final state
                visualizer.save_image(f"{run_dir}/final_state.png")
                # Pause to see final state
                time.sleep(1.0)
                break
        
        # Save the animation for this run
        animation_path = os.path.join(run_dir, f"roundup_demo_run_{run_idx+1}.mp4")
        visualizer.save_animation(animation_path, fps=5)
        print(f"Animation saved to: {animation_path}")
        
        # Log summary of this run
        if args.logging and LOGGER_AVAILABLE:
            main_logger.log_episode_end(
                episode=run_idx, 
                reward=episode_reward, 
                success=success, 
                steps=step+1
            )
        
        # Cleanup visualizer for this run
        visualizer.close()
    
    # Combine all runs into a summary if multiple runs
    if args.num_runs > 1 and args.logging and LOGGER_AVAILABLE:
        stats = main_logger.get_summary_stats()
        if isinstance(stats, dict):
            print("\nSummary of all runs:")
            print(f"Total runs: {stats['total_episodes']}")
            print(f"Success rate: {stats['success_rate']:.2f}")
            print(f"Average steps per run: {stats['avg_steps']:.2f}")
            main_logger.save_summary()
    
    print("Demo mode completed!")

if __name__ == "__main__":
    args = parse_args()
    
    # Set random seeds if provided, otherwise use time-based seeds
    if args.seed is not None:
        print(f"Using fixed random seed: {args.seed}")
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        random.seed(args.seed)
    else:
        # Use time-based seed for more randomness
        time_seed = int(time.time() * 1000) % 100000
        print(f"Using time-based random seed: {time_seed}")
        np.random.seed(time_seed)
        tf.random.set_seed(time_seed)
        random.seed(time_seed)
    
    # Create output directory if it doesn't exist
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Create logs directory if it doesn't exist and logging is enabled
    if args.logging and not os.path.exists('./logs'):
        os.makedirs('./logs')
        print("Created logs directory")
    
    # Run selected mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'demo':
        demo(args)
    else:
        print("Invalid mode. Use 'train', 'test', or 'demo'.")