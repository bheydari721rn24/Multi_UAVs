import numpy as np
from collections import deque
from utils import CONFIG


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, correlation_index):
        """Add experience to the buffer with its correlation index."""
        experience = (state, action, reward, next_state, correlation_index)
        self.buffer.append(experience)
        # Initially set equal priority for all experiences
        self.priorities.append(1.0)
    
    def sample_batch(self, batch_size):
        """Sample a batch of experiences using prioritized experience replay."""
        if len(self.buffer) < batch_size:
            return [], [], [], [], [], []
        
        # Use priorities for sampling (PER strategy)
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, correlation_indices = zip(*batch)
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(correlation_indices), indices
    
    def update_priorities(self, indices, errors, alpha=0.6):
        """Update priorities based on TD errors."""
        for i, e in zip(indices, errors):
            if i < len(self.priorities):  # Ensure index is valid
                self.priorities[i] = (abs(e) + 1e-6) ** (alpha + 0.1 * np.random.rand())  # Update alpha with a random value between 0.6 and 0.7
    
    def sample_by_correlation(self, batch_size, current_corr_index):
        """
        Sample based on correlation index similarity (REL strategy).
        First sample using PER, then select most relevant samples by correlation.
        """
        if len(self.buffer) < CONFIG["pre_batch_size"]:
            return [], [], [], [], [], []
        
        # First sample using PER
        states, actions, rewards, next_states, correlation_indices, indices = self.sample_batch(CONFIG["pre_batch_size"])
        
        # Then filter by correlation similarity (REL strategy)
        corr_diffs = np.abs(correlation_indices - current_corr_index)
        sorted_indices = np.argsort(corr_diffs)[:batch_size]
        
        final_states = states[sorted_indices]
        final_actions = actions[sorted_indices]
        final_rewards = rewards[sorted_indices]
        final_next_states = next_states[sorted_indices]
        final_corr_indices = correlation_indices[sorted_indices]
        final_buffer_indices = [indices[i] for i in sorted_indices]
        
        # Add curriculum learning strategy
        curriculum_threshold = 0.5
        if np.mean(corr_diffs) < curriculum_threshold:
            # If the average correlation difference is below the threshold, 
            # increase the batch size to include more diverse samples
            batch_size = int(batch_size * 1.5)
            sorted_indices = np.argsort(corr_diffs)[:batch_size]
            final_states = states[sorted_indices]
            final_actions = actions[sorted_indices]
            final_rewards = rewards[sorted_indices]
            final_next_states = next_states[sorted_indices]
            final_corr_indices = correlation_indices[sorted_indices]
            final_buffer_indices = [indices[i] for i in sorted_indices]
        
        # Calculate the correlation coefficient between the current correlation index and the sampled correlation indices
        
        # Fix: Ensure the shapes are compatible for np.corrcoef
        # current_corr_index is a scalar but we need an array for correlation calculation
        if np.isscalar(current_corr_index):
            current_corr_index_array = np.array([current_corr_index] * len(final_corr_indices))
        else:
            current_corr_index_array = current_corr_index
            
        # Now calculate correlation with handling for zero standard deviation
        if len(final_corr_indices) > 0:
            # Check if both arrays have non-zero standard deviation before calculating correlation
            std1 = np.std(current_corr_index_array)
            std2 = np.std(final_corr_indices)
            
            if std1 > 0 and std2 > 0:
                corr_coef = np.corrcoef(current_corr_index_array, final_corr_indices)[0, 1]
                if np.isnan(corr_coef):
                    corr_coef = 0
            else:
                # Cannot calculate meaningful correlation with zero variance
                corr_coef = 0
        else:
            corr_coef = 0  # Default value if no samples
        
        # If the correlation coefficient is high, increase the batch size to include more diverse samples
        if corr_coef > 0.7:
            batch_size = int(batch_size * 1.2)
            sorted_indices = np.argsort(corr_diffs)[:batch_size]
            final_states = states[sorted_indices]
            final_actions = actions[sorted_indices]
            final_rewards = rewards[sorted_indices]
            final_next_states = next_states[sorted_indices]
            final_corr_indices = correlation_indices[sorted_indices]
            final_buffer_indices = [indices[i] for i in sorted_indices]
        
        return final_states, final_actions, final_rewards, final_next_states, final_corr_indices, final_buffer_indices
    
    def size(self):
        """Return the current size of the buffer."""
        return len(self.buffer)