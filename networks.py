import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from utils import CONFIG

class ActorNetwork:
    def __init__(self, state_dim, action_dim, name, lr=CONFIG["actor_lr"]):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.learning_rate = lr
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration rate decay
        
        # Build the model
        self.model = self._build_model()
        self.target_model = self._build_model()
        
        # Initialize target model with same weights
        self.target_model.set_weights(self.model.get_weights())
    
    def _build_model(self):
        """Build the actor network as described in the paper."""
        inputs = Input(shape=(self.state_dim,))
        
        # Hidden layers
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        
        # Output layer with tanh activation to limit to [-1, 1]
        outputs = Dense(self.action_dim, activation='tanh')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=f"{self.name}_actor")
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def predict(self, state, add_noise=True):
        """Predict action from state, with optional exploration noise."""
        state_tensor = np.array([state]) if state.ndim == 1 else state
        action = self.model.predict(state_tensor, verbose=0)[0]
        
        if add_noise:
            # Add exploration noise
            noise = np.random.normal(0, self.epsilon, size=self.action_dim)
            action += noise
            action = np.clip(action, -1.0, 1.0)  # Clip to valid range
            
        return action
    
    def target_predict(self, state):
        """Predict action using target network."""
        state_tensor = np.array([state]) if state.ndim == 1 else state
        return self.target_model.predict(state_tensor, verbose=0)[0]
    
    def train(self, states, action_grads):
        """Train the actor network using policy gradient."""
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Generate actions
            actions = self.model(states)
            # Negative because we want to maximize Q (minimize -Q)
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
        """Soft update target network."""
        tau = CONFIG["tau"]
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        
        for i in range(len(actor_weights)):
            actor_target_weights[i] = tau * actor_weights[i] + (1 - tau) * actor_target_weights[i]
        
        self.target_model.set_weights(actor_target_weights)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon *= self.epsilon_decay
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
    def __init__(self, state_dim, action_dims, name, lr=CONFIG["critic_lr"]):
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
        """Build the critic network as described in the paper."""
        # Input for state
        state_input = Input(shape=(self.state_dim,))
        
        # Inputs for all agents' actions
        action_inputs = []
        for dim in self.action_dims:
            action_inputs.append(Input(shape=(dim,)))
        
        # Concatenate all inputs
        concat_input = Concatenate()([state_input] + action_inputs)
        
        # Hidden layers
        x = Dense(128, activation='relu')(concat_input)
        x = Dense(128, activation='relu')(x)
        
        # Output layer (Q-value)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=[state_input] + action_inputs, outputs=output, name=f"{self.name}_critic")
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return model
    
    def predict(self, state, actions):
        """Predict Q-value for state-action pair."""
        state_tensor = np.array([state]) if state.ndim == 1 else state
        action_tensors = [np.array([a]) if a.ndim == 1 else a for a in actions]
        predictions = self.model.predict([state_tensor] + action_tensors, verbose=0)
        return predictions.flatten()  # Return flattened array for easier handling
    
    def target_predict(self, state, actions):
        """Predict Q-value using target network."""
        state_tensor = np.array([state]) if state.ndim == 1 else state
        action_tensors = [np.array([a]) if a.ndim == 1 else a for a in actions]
        predictions = self.target_model.predict([state_tensor] + action_tensors, verbose=0)
        return predictions.flatten()  # Return flattened array for easier handling
    
    def train(self, states, actions, target_q_values):
        """Train the critic network using TD learning."""
        # Convert inputs to tensors
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
        """Get gradients of Q with respect to actions of a specific agent."""
        actions_variables = []
        for i, a in enumerate(actions):
            if i == agent_index:
                # Convert to tensor and make it watchable by tape
                actions_variables.append(tf.convert_to_tensor(a, dtype=tf.float32))
            else:
                # For other agents' actions, we don't need to watch them
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
        """Soft update target network."""
        tau = CONFIG["tau"]
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        
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