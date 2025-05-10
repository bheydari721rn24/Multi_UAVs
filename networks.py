# Neural networks implementation for Multi-UAV control system
# This module implements the Actor-Critic architecture for reinforcement learning

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import CONFIG  # Import configuration parameters

class ActorNetwork:
    """Actor network for the MADDPG algorithm.
    
    The actor network maps states to actions. Each UAV has its own actor network
    that determines the best action to take based on its local observations.
    This includes sensing obstacles and other UAVs to implement avoidance behaviors.
    """
    def __init__(self, state_dim, action_dim, name, lr=CONFIG["actor_lr"]):
        """Initialize the actor network.
        
        Args:
            state_dim: Dimension of the state space (UAV observations)
            action_dim: Dimension of the action space (UAV control outputs)
            name: Unique name for this actor network
            lr: Learning rate for actor network updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.learning_rate = lr
        self.epsilon = 1.0  # Initial exploration rate - high at first for environment discovery
        self.epsilon_min = 0.01  # Minimum exploration rate - maintains some exploration
        self.epsilon_decay = 0.995  # Exploration rate decay - gradually reduces exploration
        
        # Build the model
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        # Initialize target model with same weights
        self.target_model.set_weights(self.model.get_weights())
    
    def _build_model(self):
        """Build the actor network architecture.
        
        The actor network consists of:
        1. Input layer for state observations (including obstacle sensor data)
        2. Two hidden layers with ReLU activation for complex feature extraction
        3. Output layer with tanh activation to bound actions within [-1, 1]
           These normalized actions control UAV acceleration and heading
        
        Returns:
            A compiled Keras model that maps states to actions
        """
        inputs = Input(shape=(self.state_dim,))
        
        # Hidden layers for feature extraction
        x = Dense(128, activation='relu')(inputs)  # First hidden layer with 128 neurons
        x = Dense(128, activation='relu')(x)      # Second hidden layer with 128 neurons
        
        # Output layer with tanh activation to limit actions to [-1, 1] range
        # This ensures smooth, controlled movements for the UAVs
        outputs = Dense(self.action_dim, activation='tanh')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=f"{self.name}_actor")
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def predict(self, state, add_noise=True):
        """Predict UAV control actions from current state observations.
        
        This method processes sensor data (including obstacle detection) and
        outputs control signals for UAV movement. The noise parameter enables
        exploration during training to discover better obstacle avoidance strategies.
        
        Args:
            state: Current state/observation of the UAV's environment
            add_noise: Whether to add exploration noise to the action
        
        Returns:
            Action vector with normalized values in [-1, 1] range
        """
        # Convert state to tensor format required by TensorFlow
        state_tensor = np.array([state]) if state.ndim == 1 else state
        # Generate base action from current policy
        action = self.model.predict(state_tensor, verbose=0)[0]
        
        if add_noise:
            # Add Gaussian exploration noise scaled by current epsilon
            # This helps discover better obstacle avoidance strategies during training
            noise = np.random.normal(0, self.epsilon, size=self.action_dim)
            action += noise
            # Ensure actions remain within valid control range
            action = np.clip(action, -1.0, 1.0)  # Clip to valid range
            
        return action
    
    def target_predict(self, state):
        """Predict action using target network for stable learning.
        
        The target network provides a stable reference point during training,
        which helps prevent oscillations in the learning process.
        
        Args:
            state: Current state/observation of the UAV's environment
            
        Returns:
            Target action vector without exploration noise
        """
        state_tensor = np.array([state]) if state.ndim == 1 else state
        return self.target_model.predict(state_tensor, verbose=0)[0]
    
    def train(self, states, action_grads):
        """Train the actor network using policy gradient methods.
        
        Updates the actor network by following the policy gradient that maximizes expected Q-value.
        This training process improves the UAV's ability to avoid obstacles and coordinate with others.
        
        Args:
            states: Batch of state observations 
            action_grads: Gradients of critic Q-value with respect to actions
            
        Returns:
            The actor loss value for monitoring training progress
        """
        # Convert numpy arrays to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Generate actions from current policy
            actions = self.model(states)
            # Negative because we want to maximize Q (minimize -Q)
            # This encourages actions that lead to higher Q-values (better obstacle avoidance)
            actor_loss = -tf.reduce_mean(tf.reduce_sum(actions * action_grads, axis=1))
        
        # Get gradients
        actor_gradients = tape.gradient(actor_loss, self.model.trainable_variables)
        
        # Apply gradients with clipping to prevent exploding gradients
        grads_and_vars = [(tf.clip_by_norm(g, 1.0), v) for g, v in zip(actor_gradients, self.model.trainable_variables)]
        self.model.optimizer.apply_gradients(grads_and_vars)
        
        # Decay exploration rate
        self.decay_epsilon()
        
        return actor_loss
    
    def update_target(self):
        """Soft update target network weights.
        
        Instead of directly copying weights, this performs a soft update where
        the target network slowly tracks the main network. This creates more
        stable learning dynamics and prevents abrupt policy changes.
        
        The update follows: target = tau * main + (1-tau) * target
        where tau is typically a small value (e.g., 0.01)
        """
        tau = CONFIG["tau"]  # Soft update coefficient (typically small)
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        
        # Update each layer's weights using the soft update formula
        for i in range(len(actor_weights)):
            actor_target_weights[i] = tau * actor_weights[i] + (1 - tau) * actor_target_weights[i]
        
        self.target_model.set_weights(actor_target_weights)
    
    def decay_epsilon(self):
        """Decay the exploration rate over time.
        
        As training progresses, we reduce the amount of random exploration.
        This allows the UAVs to transition from initial random movements to
        more refined, learned behaviors around obstacles and other UAVs.
        
        The epsilon value will never go below epsilon_min to maintain some
        minimum level of exploration throughout training.
        """
        # Apply multiplicative decay
        self.epsilon *= self.epsilon_decay
        # Ensure epsilon doesn't go below minimum value
        self.epsilon = max(self.epsilon, self.epsilon_min)
    
    def save(self, path):
        """Save the model weights."""
        self.model.save_weights(f"{path}/{self.name}.h5")
    
    def load(self, path):
        """Load the model weights."""
        try:
            # Allow partial loading in case the architecture changed slightly.
            self.model.load_weights(f"{path}/{self.name}.h5", by_name=True, skip_mismatch=True)
            self.target_model.load_weights(f"{path}/{self.name}.h5", by_name=True, skip_mismatch=True)
            print(f"Successfully loaded {self.name} model (with skip_mismatch)")
            return True
        except Exception as e:
            print(f"Failed to load {self.name} model: {e}")
            return False

class CriticNetwork:
    """Critic network for the MADDPG algorithm.
    
    The critic evaluates the quality of actions taken by all UAVs collectively.
    It estimates the Q-value of a state-action pair, considering the interactions
    between UAVs, their environment, and obstacles for effective coordination.
    """
    def __init__(self, state_dim, action_dims, name, lr=CONFIG["critic_lr"]):
        """Initialize the critic network.
        
        Args:
            state_dim: Dimension of the state space (global environment state)
            action_dims: List of action dimensions for each UAV agent
            name: Unique name for this critic network
            lr: Learning rate for critic network updates
        """
        self.state_dim = state_dim
        self.action_dims = action_dims  # List of action dimensions for each agent
        self.name = name
        self.learning_rate = lr
        
        # Build the model
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        # Initialize target model with same weights
        self.target_model.set_weights(self.model.get_weights())
    
    def _build_model(self):
        """Build the critic network architecture.
        
        The critic network consists of:
        1. Input layer for global state (including positions of all UAVs, target, and obstacles)
        2. Separate input layers for each UAV's actions
        3. Concatenation of all inputs to process the collective behavior
        4. Two hidden layers to model complex interactions between UAVs and obstacles
        5. Linear output layer that produces a single Q-value 
           (higher values indicate better collective behavior)
        
        Returns:
            A compiled Keras model that estimates the Q-value for the given state-actions
        """
        # Input for global state information
        state_input = Input(shape=(self.state_dim,))
        
        # Separate inputs for each UAV's actions
        # This allows the critic to evaluate how well UAVs coordinate their movements
        action_inputs = []
        for dim in self.action_dims:
            action_inputs.append(Input(shape=(dim,)))
        
        # Concatenate all inputs to process collective behavior
        concat_input = Concatenate()([state_input] + action_inputs)
        
        # Hidden layers to model complex interactions
        x = Dense(128, activation='relu')(concat_input)  # First hidden layer
        x = Dense(128, activation='relu')(x)            # Second hidden layer
        
        # Output layer producing a single Q-value
        # Higher Q-values represent better collective obstacle avoidance and coordination
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input] + action_inputs, outputs=output, name=f"{self.name}_critic")
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return model
    
    def predict(self, state, actions):
        """Predict Q-value for a given state and collective UAV actions.
        
        This evaluates how good the current actions of all UAVs are together,
        considering obstacle avoidance, target tracking, and coordination.
        
        Args:
            state: Global state observation including all UAVs, target, and obstacles
            actions: List of actions for all UAVs
            
        Returns:
            Estimated Q-value (higher is better) for the collective behavior
        """
        # Convert inputs to tensor format
        state_tensor = np.array([state]) if state.ndim == 1 else state
        batch_size = state_tensor.shape[0]
        
        # Ensure all action tensors have the same batch dimension as state_tensor
        action_tensors = []
        for a in actions:
            if a.ndim == 1:
                # Single action, needs to be repeated for the batch
                a_expanded = np.expand_dims(a, axis=0)
                if batch_size > 1:
                    a_expanded = np.repeat(a_expanded, batch_size, axis=0)
                action_tensors.append(a_expanded)
            else:
                # Batch of actions - ensure same batch size
                if a.shape[0] != batch_size:
                    raise ValueError(f"Action batch size {a.shape[0]} doesn't match state batch size {batch_size}")
                action_tensors.append(a)
        
        # Generate Q-value prediction
        predictions = self.model.predict([state_tensor] + action_tensors, verbose=0)
        return predictions.flatten()  # Return flattened array for easier handling
    
    def target_predict(self, state, actions):
        """Predict Q-value using the stable target network.
        
        Similar to predict(), but uses the more stable target network which
        updates more slowly, providing stability in the learning process.
        
        Args:
            state: Global state observation
            actions: List of actions for all UAVs
            
        Returns:
            Target Q-value estimate for the collective behavior
        """
        state_tensor = np.array([state]) if state.ndim == 1 else state
        batch_size = state_tensor.shape[0]
        
        # Ensure all action tensors have the same batch dimension as state_tensor
        action_tensors = []
        for a in actions:
            if a.ndim == 1:
                # Single action, needs to be repeated for the batch
                a_expanded = np.expand_dims(a, axis=0)
                if batch_size > 1:
                    a_expanded = np.repeat(a_expanded, batch_size, axis=0)
                action_tensors.append(a_expanded)
            else:
                # Batch of actions - ensure same batch size
                if a.shape[0] != batch_size:
                    raise ValueError(f"Action batch size {a.shape[0]} doesn't match state batch size {batch_size}")
                action_tensors.append(a)
        
        predictions = self.target_model.predict([state_tensor] + action_tensors, verbose=0)
        return predictions.flatten()  # Return flattened array for easier handling

    def train(self, states, actions, target_q_values):
        """Train the critic network using Temporal Difference (TD) learning.
        
        Updates the critic's estimates of Q-values based on observed rewards
        and estimated future rewards. This improves the critic's ability to
        evaluate UAV actions considering obstacle avoidance and coordination.
        
        Args:
            states: Batch of state observations
            actions: List of action batches for all UAVs
            target_q_values: Target Q-values calculated using rewards and future estimates
            
        Returns:
            The critic loss value for monitoring training progress
        """
        # Convert inputs to tensors for TensorFlow operations
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensors = [tf.convert_to_tensor(a, dtype=tf.float32) for a in actions]
        target_q_values = tf.convert_to_tensor(target_q_values, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Get predicted Q values from critic network
            q_values = self.model([states] + actions_tensors)
            # Reshape target to match q_values if needed
            target_q_values = tf.reshape(target_q_values, q_values.shape)
            # MSE loss
            critic_loss = tf.reduce_mean(tf.square(target_q_values - q_values))
        
        # Get gradients
        critic_gradients = tape.gradient(critic_loss, self.model.trainable_variables)
        
        # Apply gradients with clipping
        grads_and_vars = [(tf.clip_by_norm(g, 1.0), v) for g, v in zip(critic_gradients, self.model.trainable_variables)]
        self.model.optimizer.apply_gradients(grads_and_vars)
        
        return critic_loss
    
    def get_action_gradients(self, states, actions, agent_index):
        """Get gradients of Q-value with respect to a specific UAV's actions.
        
        This calculates how changes in one UAV's actions would affect the overall
        Q-value, which is essential for updating the actor network. These gradients
        guide the UAV toward actions that improve collective performance, including
        better obstacle avoidance and coordination.
        
        Args:
            states: Batch of state observations
            actions: List of action batches for all UAVs
            agent_index: Index of the UAV whose action gradients we want
            
        Returns:
            Gradients of Q-value with respect to the specified UAV's actions
        """
        # Prepare action variables with different handling for the target agent
        actions_variables = []
        for i, a in enumerate(actions):
            if i == agent_index:
                # Convert to tensor and make it watchable by tape for gradient calculation
                actions_variables.append(tf.convert_to_tensor(a, dtype=tf.float32))
            else:
                # For other agents' actions, we don't need to watch them
                # as we're only interested in the target agent's gradients
                actions_variables.append(tf.convert_to_tensor(a, dtype=tf.float32, name=f"agent_{i}_actions"))
        
        with tf.GradientTape() as tape:
            # Only watch the actions of the agent we're updating
            tape.watch(actions_variables[agent_index])
            q_values = self.model([states] + actions_variables)
            # Take mean across batch dimension
            q_mean = tf.reduce_mean(q_values)
        
        # Get gradient of mean Q with respect to actions
        action_gradients = tape.gradient(q_mean, actions_variables[agent_index])
        return action_gradients
    
    def update_target(self):
        """Soft update target critic network weights.
        
        Gradually updates the target network to track the main network,
        maintaining stability in learning while still adapting to new information.
        This is especially important in multi-agent scenarios with obstacles
        where the environment complexity can cause learning instability.
        
        Uses the same tau parameter as the actor's target update.
        """
        tau = CONFIG["tau"]  # Soft update coefficient
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        
        # Apply soft update formula to each layer's weights
        for i in range(len(critic_weights)):
            critic_target_weights[i] = tau * critic_weights[i] + (1 - tau) * critic_target_weights[i]
        
        self.target_model.set_weights(critic_target_weights)
    
    def save(self, path):
        """Save the model weights."""
        self.model.save_weights(f"{path}/{self.name}.h5")
    
    def load(self, path):
        """Load the model weights."""
        try:
            # Allow partial loading in case the architecture changed slightly.
            self.model.load_weights(f"{path}/{self.name}.h5", by_name=True, skip_mismatch=True)
            self.target_model.load_weights(f"{path}/{self.name}.h5", by_name=True, skip_mismatch=True)
            print(f"Successfully loaded {self.name} model (with skip_mismatch)")
            return True
        except Exception as e:
            print(f"Failed to load {self.name} model: {e}")
            return False