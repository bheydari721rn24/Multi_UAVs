import numpy as np
import tensorflow as tf
import os
from environment import Environment
from cel_maddpg import CEL_MADDPG
from utils import CONFIG

# Simplify debug for neural network training verification

# Initial settings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create simulation environment
env = Environment(
    num_uavs=3,
    num_obstacles=3,
    dynamic_obstacles=True
)

# Initialize CEL-MADDPG algorithm
state_dim = env.get_state_dim()
action_dim = 2  # [Fx, Fy]
agent = CEL_MADDPG(
    state_dim=state_dim,
    action_dim=action_dim,
    num_agents=3,
    save_path="./debug_models"
)

# Run a limited number of episodes to verify reward changes
for episode in range(10):  # Increased number of episodes
    state = env.reset()
    total_reward = 0
    
    print(f"\n--- Episode {episode} ---")
    print(f"Buffer size: {agent.replay_buffer.size()}, Required: {CONFIG['pre_batch_size']}")
    print(f"Curriculum Learning: step={agent.curriculum_current_step}, threshold={agent.curriculum_threshold}")
    
    for step in range(CONFIG["max_steps_per_episode"]):
        # Get actions from current policy
        actions = agent.get_actions(state)
        
        # Execute actions in environment
        next_state, rewards, done, info = env.step(actions)
        
        # Calculate correlation index for current state
        corr_index = agent.calculate_correlation_index(env)
        
        # Store experience in memory
        agent.store_experience(state, actions, rewards, next_state, corr_index)
        
        # Train if sufficient samples are available
        if agent.replay_buffer.size() >= CONFIG["pre_batch_size"]:
            print(f"  Training at step {step}")
            
            # Store weights before training
            try:
                before_actor_weights = agent.actors[0].model.get_weights()[0][0][:3]
                before_critic_weights = agent.critics[0].model.get_weights()[0][0][:3]
                print(f"  Before - Actor weights: {before_actor_weights}")
                print(f"  Before - Critic weights: {before_critic_weights}")
            except Exception as e:
                print(f"  Could not get weights before training: {e}")
            
            # Perform training
            agent.train(corr_index)
            
            # Check model weights after training
            try:
                after_actor_weights = agent.actors[0].model.get_weights()[0][0][:3]
                after_critic_weights = agent.critics[0].model.get_weights()[0][0][:3]
                print(f"  After - Actor weights: {after_actor_weights}")
                print(f"  After - Critic weights: {after_critic_weights}")
                
                # Calculate and display changes
                actor_change = np.abs(np.sum(after_actor_weights - before_actor_weights))
                critic_change = np.abs(np.sum(after_critic_weights - before_critic_weights))
                print(f"  Weight changes - Actor: {actor_change:.8f}, Critic: {critic_change:.8f}\n")
            except Exception as e:
                print(f"  Could not get weights after training: {e}\n")
        
        # Update state and reward
        state = next_state
        total_reward += sum(rewards)
        
        if done:
            break
            
    # Display episode end information
    success = 1 if info.get("success", False) else 0
    task_state = info.get("task_state", "unknown")
    print(f"Episode {episode} - Total reward: {total_reward:.2f}, Task: {task_state}, Success: {success}")
    
    # Display distance to target for status verification
    distances = [np.linalg.norm(uav.position - env.target.position) for uav in env.uavs]
    print(f"Final distances to target: {[round(d, 2) for d in distances]}")

print("\nDone!")
