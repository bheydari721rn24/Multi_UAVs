"""
Federated Learning Module for Multi-UAV Systems

This module implements federated knowledge sharing between UAVs,
allowing decentralized learning without a central coordinator.

Author: Research Team
Date: 2025
"""

import numpy as np
from utils import CONFIG


class FederatedLearner:
    """
    Federated learning system for decentralized knowledge sharing
    Enables UAVs to exchange model parameters without central coordination
    """
    
    def __init__(self, uav_id, knowledge_dim=None):
        """Initialize federated learner with configuration parameters"""
        self.config = CONFIG["swarm"]["federated_learning"]
        
        self.uav_id = uav_id
        
        if knowledge_dim is None:
            self.knowledge_dim = self.config["knowledge_dimension"]
        else:
            self.knowledge_dim = knowledge_dim
            
        self.communication_range = self.config["communication_range"]
        self.sharing_interval = self.config["sharing_interval"]
        self.model_fusion_weight = self.config["model_fusion_weight"]
        self.confidence_threshold = self.config["fusion_confidence_threshold"]
        self.federation_topology = self.config["federation_topology"]
        
        # Initialize local knowledge tensor
        self.knowledge_tensor = np.zeros(self.knowledge_dim)
        
        # Initialize local model parameters
        self.local_parameters = {}
        
        # Track received messages
        self.received_messages = []
        
        # Communication history
        self.communication_history = []
        
        # Step counter
        self.step_count = 0
        
        # Knowledge confidence (0-1)
        self.knowledge_confidence = 0.5
        
    def update_knowledge(self, observation, reward, confidence=0.5):
        """
        Update local knowledge tensor with new observation
        
        Args:
            observation: Observation vector
            reward: Reward value
            confidence: Confidence in observation
            
        Returns:
            numpy.ndarray: Updated knowledge tensor
        """
        if observation is None:
            return self.knowledge_tensor
            
        # Convert observation to proper dimensions if needed
        if hasattr(observation, '__len__'):
            observation_vector = np.array(observation)
            if len(observation_vector) > self.knowledge_dim:
                # Truncate if too large
                observation_vector = observation_vector[:self.knowledge_dim]
            elif len(observation_vector) < self.knowledge_dim:
                # Pad if too small
                padding = np.zeros(self.knowledge_dim - len(observation_vector))
                observation_vector = np.concatenate([observation_vector, padding])
        else:
            # Scalar observation
            observation_vector = np.zeros(self.knowledge_dim)
            observation_vector[0] = observation
            
        # Scale observation by reward (more impactful if reward is high)
        impact = 0.1 + 0.4 * max(0, reward)
        
        # Update knowledge tensor as weighted average
        # Higher confidence gives more weight to new observation
        self.knowledge_tensor = (1 - confidence * impact) * self.knowledge_tensor + \
                               confidence * impact * observation_vector
                               
        # Update overall knowledge confidence
        self.knowledge_confidence = max(self.knowledge_confidence, confidence)
        
        return self.knowledge_tensor
        
    def update_model_parameter(self, parameter_name, parameter_value, confidence=0.5):
        """
        Update local model parameter
        
        Args:
            parameter_name: Name of parameter
            parameter_value: New parameter value
            confidence: Confidence in parameter update
            
        Returns:
            float: Updated parameter value
        """
        if parameter_name not in self.local_parameters:
            # New parameter
            self.local_parameters[parameter_name] = {
                "value": parameter_value,
                "confidence": confidence,
                "update_count": 1
            }
            return parameter_value
            
        # Existing parameter - update with weighted average
        current = self.local_parameters[parameter_name]
        current_value = current["value"]
        current_confidence = current["confidence"]
        
        # Higher confidence gives more weight to new value
        weight = confidence / (current_confidence + confidence)
        
        # Update parameter
        new_value = (1 - weight) * current_value + weight * parameter_value
        
        # Update metadata
        self.local_parameters[parameter_name] = {
            "value": new_value,
            "confidence": max(current_confidence, confidence),
            "update_count": current["update_count"] + 1
        }
        
        return new_value
        
    def share_knowledge(self, nearby_uavs):
        """
        Share knowledge with nearby UAVs
        
        Args:
            nearby_uavs: List of nearby UAVs
            
        Returns:
            list: List of messages shared
        """
        self.step_count += 1
        
        # Only share on specified intervals
        if self.step_count % self.sharing_interval != 0:
            return []
            
        # Create knowledge message
        message = {
            "sender_id": self.uav_id,
            "knowledge_tensor": self.knowledge_tensor,
            "confidence": self.knowledge_confidence,
            "timestamp": self.step_count,
            "parameters": {
                k: v for k, v in self.local_parameters.items() 
                if v["confidence"] >= self.confidence_threshold
            }
        }
        
        # Determine recipients based on federation topology
        recipients = []
        
        if self.federation_topology == "dynamic":
            # Share with all UAVs within communication range
            for uav in nearby_uavs:
                recipients.append(uav)
                
        elif self.federation_topology == "ring":
            # Share only with next UAV in ring
            if nearby_uavs:
                # Sort by ID
                sorted_uavs = sorted(nearby_uavs, key=lambda u: u.id)
                
                # Find own position in sorted list
                own_idx = next((i for i, u in enumerate(sorted_uavs) if u.id > self.uav_id), 0)
                
                # Share with next UAV in ring
                next_idx = (own_idx + 1) % len(sorted_uavs)
                recipients.append(sorted_uavs[next_idx])
                
        elif self.federation_topology == "star":
            # Star topology: central node shares with all, others share only with center
            # Determine center (lowest ID)
            if nearby_uavs:
                center_uav = min(nearby_uavs + [self], key=lambda u: u.id)
                
                if center_uav.id == self.uav_id:
                    # This UAV is center, share with all
                    recipients = nearby_uavs
                else:
                    # Not center, share only with center
                    recipients = [center_uav]
                    
        else:  # "mesh" or default
            # Share with all UAVs
            recipients = nearby_uavs
            
        # Share message with recipients
        shared_messages = []
        
        for recipient in recipients:
            # Check if within communication range
            distance = np.linalg.norm(
                getattr(recipient, 'position', np.zeros(2)) - 
                getattr(self, 'position', np.zeros(2))
            )
            
            if distance <= self.communication_range:
                # Record communication
                communication_record = {
                    "from_id": self.uav_id,
                    "to_id": recipient.id,
                    "timestamp": self.step_count,
                    "success": True
                }
                
                self.communication_history.append(communication_record)
                
                # Recipient receives message
                if hasattr(recipient, 'receive_message'):
                    recipient.receive_message(message)
                    
                shared_messages.append((recipient.id, message))
                
        return shared_messages
        
    def receive_message(self, message):
        """
        Receive knowledge message from another UAV
        
        Args:
            message: Knowledge message
            
        Returns:
            bool: True if message accepted
        """
        # Store message
        self.received_messages.append(message)
        
        # Extract message components
        sender_id = message.get("sender_id")
        knowledge_tensor = message.get("knowledge_tensor")
        confidence = message.get("confidence", 0.5)
        parameters = message.get("parameters", {})
        
        # Only process message if confidence meets threshold
        if confidence < self.confidence_threshold:
            return False
            
        # Fuse knowledge tensor
        if knowledge_tensor is not None and len(knowledge_tensor) == self.knowledge_dim:
            # Weight for fusion depends on relative confidence
            fusion_weight = self.model_fusion_weight * (confidence / max(1.0, self.knowledge_confidence + confidence))
            
            # Fuse knowledge tensors
            self.knowledge_tensor = (1 - fusion_weight) * self.knowledge_tensor + \
                                   fusion_weight * knowledge_tensor
                                   
            # Update confidence
            self.knowledge_confidence = max(self.knowledge_confidence, confidence * 0.9)
            
        # Fuse model parameters
        for param_name, param_data in parameters.items():
            if isinstance(param_data, dict):
                param_value = param_data.get("value")
                param_confidence = param_data.get("confidence", 0.5)
                
                if param_value is not None:
                    self.update_model_parameter(param_name, param_value, param_confidence * 0.9)
                    
        return True
        
    def process_messages(self):
        """
        Process all received messages
        
        Returns:
            int: Number of messages processed
        """
        if not self.received_messages:
            return 0
            
        # Process each message
        processed_count = 0
        
        for message in self.received_messages:
            if self.receive_message(message):
                processed_count += 1
                
        # Clear messages
        self.received_messages = []
        
        return processed_count
        
    def aggregate_knowledge(self, nearby_uavs):
        """
        Aggregate knowledge from nearby UAVs
        
        Args:
            nearby_uavs: List of nearby UAVs
            
        Returns:
            dict: Aggregated knowledge
        """
        # Collect knowledge from nearby UAVs
        all_knowledge = [self.knowledge_tensor]
        all_confidences = [self.knowledge_confidence]
        
        for uav in nearby_uavs:
            if hasattr(uav, 'federated_learner'):
                all_knowledge.append(uav.federated_learner.knowledge_tensor)
                all_confidences.append(uav.federated_learner.knowledge_confidence)
                
        # Convert to numpy arrays
        knowledge_array = np.array(all_knowledge)
        confidence_array = np.array(all_confidences)
        
        # Normalize confidences to weights
        total_confidence = np.sum(confidence_array)
        if total_confidence > 0:
            weights = confidence_array / total_confidence
        else:
            weights = np.ones_like(confidence_array) / len(confidence_array)
            
        # Weighted average of knowledge tensors
        aggregated_tensor = np.zeros_like(self.knowledge_tensor)
        for i, tensor in enumerate(all_knowledge):
            aggregated_tensor += tensor * weights[i]
            
        # Collect parameters from nearby UAVs
        all_parameters = {}
        
        # Start with own parameters
        for param_name, param_data in self.local_parameters.items():
            if param_data["confidence"] >= self.confidence_threshold:
                if param_name not in all_parameters:
                    all_parameters[param_name] = []
                    
                all_parameters[param_name].append((
                    param_data["value"], param_data["confidence"]
                ))
                
        # Add parameters from nearby UAVs
        for uav in nearby_uavs:
            if hasattr(uav, 'federated_learner'):
                for param_name, param_data in uav.federated_learner.local_parameters.items():
                    if param_data["confidence"] >= self.confidence_threshold:
                        if param_name not in all_parameters:
                            all_parameters[param_name] = []
                            
                        all_parameters[param_name].append((
                            param_data["value"], param_data["confidence"]
                        ))
                        
        # Aggregate parameters
        aggregated_parameters = {}
        
        for param_name, param_values in all_parameters.items():
            # Extract values and confidences
            values = [v[0] for v in param_values]
            confidences = [v[1] for v in param_values]
            
            # Normalize confidences to weights
            total_confidence = sum(confidences)
            if total_confidence > 0:
                weights = [c / total_confidence for c in confidences]
            else:
                weights = [1.0 / len(confidences)] * len(confidences)
                
            # Weighted average
            aggregated_value = sum(v * w for v, w in zip(values, weights))
            
            # Store aggregated parameter
            aggregated_parameters[param_name] = {
                "value": aggregated_value,
                "confidence": max(confidences)
            }
            
        return {
            "knowledge_tensor": aggregated_tensor,
            "parameters": aggregated_parameters,
            "confidence": max(all_confidences)
        }
        
    def calculate_federated_force(self, uav, nearby_uavs):
        """
        Calculate force based on federated knowledge
        
        Args:
            uav: The UAV to calculate force for
            nearby_uavs: List of nearby UAVs
            
        Returns:
            numpy.ndarray: The federated force vector [x, y]
        """
        # Share knowledge with nearby UAVs
        self.share_knowledge(nearby_uavs)
        
        # Process received messages
        self.process_messages()
        
        # Aggregate knowledge
        aggregated = self.aggregate_knowledge(nearby_uavs)
        
        # Calculate force based on aggregated knowledge
        # This is a simplified implementation - in reality, would depend on
        # specific meaning of knowledge tensor
        
        # For demonstration, interpret first 2 components as force direction
        if aggregated["confidence"] > 0.3 and len(aggregated["knowledge_tensor"]) >= 2:
            direction = aggregated["knowledge_tensor"][:2]
            
            # Normalize direction
            magnitude = np.linalg.norm(direction)
            if magnitude > 0:
                direction = direction / magnitude
                
            # Scale by confidence
            force = direction * CONFIG["uav_max_acceleration"] * aggregated["confidence"]
        else:
            # Default force - small random movement
            force = np.random.randn(2) * 0.1 * CONFIG["uav_max_acceleration"]
            
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            force = force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return force
