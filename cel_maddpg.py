import numpy as np
import tensorflow as tf
import os
from networks import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer
from utils import CONFIG, calculate_triangle_area, create_directory

class CEL_MADDPG:
    def __init__(self, state_dim, action_dim, num_agents, save_path="./saved_models", load_path=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.save_path = save_path
        self.load_path = load_path
        
        # Create directory if it doesn't exist
        if save_path:
            create_directory(save_path)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(CONFIG["replay_buffer_size"])
        
        # Initialize actor and critic networks for each agent
        self.actors = []
        self.critics = []
        
        # Variables for Curriculum Learning
        self.curriculum_threshold = CONFIG["curriculum_threshold"]
        self.curriculum_step_size = CONFIG["curriculum_step_size"]
        self.curriculum_current_step = 0
        
        for i in range(num_agents):
            # Actor networks
            actor = ActorNetwork(
                state_dim=state_dim,
                action_dim=action_dim,
                name=f"agent{i}_actor",
                lr=CONFIG["actor_lr"]
            )
            self.actors.append(actor)
            
            # Critic networks (each critic takes all agents' actions)
            action_dims = [action_dim] * num_agents
            critic = CriticNetwork(
                state_dim=state_dim,
                action_dims=action_dims,
                name=f"agent{i}_critic",
                lr=CONFIG["critic_lr"]
            )
            self.critics.append(critic)
        
        # Load models if path is provided
        if load_path:
            self.load_models()
    
    def calculate_correlation_index(self, env):
        """
        Calculate the correlation index fr(st) for the current state.
        This is based on equation (29) in the paper.
        """
        # Get UAV positions and target position
        uav_positions = np.array([uav.position for uav in env.uavs])
        target_position = env.target.position
        
        # Calculate triangle areas and total encirclement area
        triangle_areas = []
        for i in range(len(uav_positions)):
            next_i = (i + 1) % len(uav_positions)
            area = calculate_triangle_area(
                uav_positions[i],
                uav_positions[next_i],
                target_position
            )
            triangle_areas.append(area)
        
        # Calculate total area of the encirclement (convex hull)
        total_encirclement_area = np.sum(triangle_areas)
        
        # Calculate distances to center (average position of UAVs)
        center = np.mean(uav_positions, axis=0)
        distances_to_center = 0
        for pos in uav_positions:
            distances_to_center += np.linalg.norm(pos - center)
        
        # Weights from CONFIG
        sigma1 = CONFIG["correlation_weights"]["sigma1"]
        sigma2 = CONFIG["correlation_weights"]["sigma2"]
        sigma3 = CONFIG["correlation_weights"]["sigma3"]
        
        # First term: difference between sum of triangles and total area (indicates if target is inside)
        term1 = sigma1 * abs(sum(triangle_areas) - total_encirclement_area)
        
        # Second term: sum of triangle areas (represents overall encirclement size)
        term2 = sigma2 * sum(triangle_areas)
        
        # Third term: distances to center (represents spread of UAVs)
        term3 = sigma3 * distances_to_center
        
        correlation_index = term1 + term2 + term3
        
        return correlation_index
    
    def get_actions(self, state, add_noise=True):
        """Get actions for all agents given the current state."""
        actions = []
        
        for i in range(self.num_agents):
            action = self.actors[i].predict(state, add_noise=add_noise)
            actions.append(action)
        
        return actions
    
    def store_experience(self, state, actions, rewards, next_state, correlation_index):
        """Store experience in replay buffer with correlation index."""
        self.replay_buffer.add(state, actions, rewards, next_state, correlation_index)
    
    def train(self, current_corr_index):
        """Train all networks using CEL-MADDPG."""
        # Sample batch using Relative Experience Learning
        states, actions, rewards, next_states, corr_indices, buffer_indices = \
            self.replay_buffer.sample_by_correlation(CONFIG["batch_size"], current_corr_index)
        
        if len(states) == 0:
            return
        
        # Update curriculum step if correlation index exceeds threshold
        if current_corr_index > self.curriculum_threshold:
            self.curriculum_current_step += self.curriculum_step_size
            self.curriculum_threshold = min(self.curriculum_threshold + self.curriculum_step_size, 1.0)
            print(f"Curriculum threshold updated: {self.curriculum_threshold}")
        
        # Train each agent's critic and actor networks
        for agent_idx in range(self.num_agents):
            # Get target actions for next state - fixed to properly use all next_states
            target_actions = []
            for i in range(self.num_agents):
                # Each agent predicts action for all next states, not just the first one
                target_action = np.array([self.actors[i].target_predict(ns) for ns in next_states])
                target_actions.append(target_action)
            
            # Calculate target Q-values using target networks - vectorized computation
            target_q_next = np.zeros(len(next_states))
            
            # Convert actions format for critic input
            for s_idx in range(len(next_states)):
                actions_for_target = [target_actions[i][s_idx] for i in range(self.num_agents)]
                target_q_next[s_idx] = self.critics[agent_idx].target_predict(next_states[s_idx], actions_for_target)
                
            # Calculate target Q value with reward and discounted future reward
            target_q = rewards[:, agent_idx] + CONFIG["gamma"] * target_q_next
            
            # Reshape actions for critic input
            actions_for_critic = [actions[:, i, :] for i in range(self.num_agents)]
            
            # Update critic
            critic_loss = self.critics[agent_idx].train(states, actions_for_critic, target_q)
            
            # First get the gradients of Q w.r.t. actions using reshaped actions
            action_gradients = self.critics[agent_idx].get_action_gradients(states, actions_for_critic, agent_idx)
            
            # Then update actor policy
            actor_loss = self.actors[agent_idx].train(states, action_gradients)
            
            # Update target networks
            self.actors[agent_idx].update_target()
            self.critics[agent_idx].update_target()
            
            # Compute predicted Q-values for computing TD errors
            predicted_q = self.critics[agent_idx].predict(states, actions_for_critic)
            td_errors = np.abs(target_q - predicted_q)
            
            # Update priorities in the replay buffer
            self.replay_buffer.update_priorities(buffer_indices, td_errors)
            
            # Don't manually apply epsilon decay here since we're now doing it in ActorNetwork.train
            # self.actors[agent_idx].epsilon *= CONFIG["epsilon_decay"]
            # self.actors[agent_idx].epsilon = max(self.actors[agent_idx].epsilon, CONFIG["epsilon_min"])
    
    def save_models(self, suffix=""):
        """Save all actor and critic models."""
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        for i in range(self.num_agents):
            # Create actor and critic subdirectories
            actor_path = f"{self.save_path}/actor{i}{suffix}"
            critic_path = f"{self.save_path}/critic{i}{suffix}"
            
            if not os.path.exists(actor_path):
                os.makedirs(actor_path)
            if not os.path.exists(critic_path):
                os.makedirs(critic_path)
            
            self.actors[i].save(actor_path)
            self.critics[i].save(critic_path)
    
    def load_models(self):
        """Load all actor and critic models."""
        if self.load_path:
            # Check if directories exist before attempting to load
            import os
            
            # First check if main directory exists
            if not os.path.exists(self.load_path):
                print(f"Model path {self.load_path} does not exist. Starting with fresh models.")
                return False
                
            all_loaded = True
            for i in range(self.num_agents):
                actor_path = f"{self.load_path}/actor{i}_final"
                critic_path = f"{self.load_path}/critic{i}_final"
                
                # Only attempt to load if both directories exist
                if os.path.exists(actor_path) and os.path.exists(critic_path):
                    actor_loaded = self.actors[i].load(actor_path)
                    critic_loaded = self.critics[i].load(critic_path)
                    all_loaded = all_loaded and actor_loaded and critic_loaded
                else:
                    print(f"Models for agent {i} not found. Starting with fresh models.")
                    all_loaded = False
            
            return all_loaded
        return False