"""
Quantum-Inspired Optimization Module for Multi-UAV Systems

This module implements quantum-inspired optimization algorithms for swarm coordination,
applying principles from quantum computing to classical swarm intelligence.

Author: Research Team
Date: 2025
"""

import cmath
import numpy as np
import time # Required for performance optimization
import math
from utils import CONFIG


class QuantumOptimizer:
    """
    Quantum-inspired optimization for swarm coordination
    Applies quantum computing principles to enhance classical swarm intelligence
    """
    
    def __init__(self):
        """Initialize quantum optimizer with configuration parameters"""
        self.config = CONFIG["swarm"]["quantum_optimization"]
        self.interference_factor = self.config["interference_factor"]
        self.phase_shift_rate = self.config["phase_shift_rate"]
        self.superposition_decay = self.config["superposition_decay"]
        self.entanglement_strength = self.config["entanglement_strength"]
        self.quantum_sigma = self.config["quantum_sigma"]
        
        # Internal quantum state
        self.quantum_state = None
        self.phase_angles = None
        self.last_update_time = 0
        
        # Performance optimization - cache calculations
        self.cached_results = {}
        self.cache_lifetime = CONFIG.get("quantum_cache_lifetime", 5)  # Default 5 time units before recalculation
        
        # Performance mode - when True, uses approximate calculations
        self.performance_mode = CONFIG.get("enable_performance_mode", True)
        self.calculation_interval = CONFIG.get("quantum_calculation_interval", 10)  # Calculate every N frames
        self.last_calculation_step = 0
        
    def optimize_action(self, swarm_force, rl_force):
        """
        Optimize action using quantum-inspired interference between force vectors
        
        Args:
            swarm_force: Force vector from swarm behaviors
            rl_force: Force vector from reinforcement learning
            
        Returns:
            numpy.ndarray: Optimized force vector
        """
        # Convert classical forces to "quantum" representation
        # This is a quantum-inspired approach, not actual quantum computing
        
        # Calculate magnitudes and angles of force vectors
        swarm_mag = np.linalg.norm(swarm_force)
        rl_mag = np.linalg.norm(rl_force)
        
        # If either force is zero, return the other one
        if swarm_mag < 1e-6:
            return rl_force
        if rl_mag < 1e-6:
            return swarm_force
            
        # Get angles (phases) of the forces
        swarm_angle = math.atan2(swarm_force[1], swarm_force[0])
        rl_angle = math.atan2(rl_force[1], rl_force[0])
        
        # Convert to complex numbers for quantum-inspired calculation
        swarm_complex = cmath.rect(swarm_mag, swarm_angle)
        rl_complex = cmath.rect(rl_mag, rl_angle)
        
        # Apply quantum interference using interference factor
        # A higher interference factor means forces will combine more coherently
        interference = self.interference_factor
        
        # Apply phase shift to create constructive/destructive interference
        phase_shift = self.phase_shift_rate * np.random.random()
        rl_complex *= cmath.exp(1j * phase_shift)
        
        # Combine using quantum superposition principles
        # This weighted combination simulates quantum interference effects
        combined_complex = swarm_complex + interference * rl_complex
        
        # Extract magnitude and phase from combined complex number
        combined_mag = abs(combined_complex)
        combined_angle = cmath.phase(combined_complex)
        
        # Convert back to Cartesian coordinates
        optimized_force = np.array([
            combined_mag * math.cos(combined_angle),
            combined_mag * math.sin(combined_angle)
        ])
        
        # Scale to ensure it doesn't exceed maximum acceleration
        optimized_mag = np.linalg.norm(optimized_force)
        if optimized_mag > CONFIG["uav_max_acceleration"]:
            optimized_force = optimized_force / optimized_mag * CONFIG["uav_max_acceleration"]
            
        return optimized_force
        
    def quantum_position_encoding(self, uav, nearby_uavs):
        """
        Encode classical position into quantum state representation
        
        Args:
            uav: The UAV for which to calculate the encoding
            nearby_uavs: List of nearby UAVs
            
        Returns:
            tuple: (amplitudes, phases) representing quantum state
        """
        if not nearby_uavs:
            return [1.0], [0.0]
            
        # Extract positions and velocities
        positions = [other_uav.position for other_uav in nearby_uavs]
        velocities = [other_uav.velocity for other_uav in nearby_uavs]
        
        # Calculate quantum amplitudes based on proximity
        amplitudes = []
        for pos in positions:
            distance = np.linalg.norm(uav.position - pos)
            # Gaussian amplitude based on distance
            amplitude = np.exp(-distance**2 / self.quantum_sigma**2)
            amplitudes.append(amplitude)
            
        # Calculate quantum phases based on velocity alignment
        phases = []
        for vel in velocities:
            vel_alignment = np.dot(uav.velocity, vel) / (
                np.linalg.norm(uav.velocity) * np.linalg.norm(vel) + 1e-8)
            # Convert alignment to phase angle
            phase = np.arccos(np.clip(vel_alignment, -1.0, 1.0))
            phases.append(phase)
            
        # Normalize amplitudes to create valid quantum state
        total = sum(amplitudes) + 1e-10
        normalized_amplitudes = [amp/total for amp in amplitudes]
        
        return normalized_amplitudes, phases
        
    def apply_quantum_interference(self, uav, nearby_uavs):
        """
        Apply quantum interference effects between UAVs
        
        Args:
            uav: Current UAV
            nearby_uavs: List of nearby UAVs
            
        Returns:
            numpy.ndarray: Interference force vector
        """
        # Initialize interference force
        interference_force = np.zeros(2)
        
        if len(nearby_uavs) < 2:
            return interference_force
            
        # Ensure quantum state and phase angles are initialized
        if self.quantum_state is None or self.phase_angles is None or len(self.quantum_state) < len(nearby_uavs):
            # Initialize with appropriate size arrays
            self.quantum_state = np.ones(len(nearby_uavs)) / len(nearby_uavs)  # Equal distribution
            self.phase_angles = np.random.uniform(0, 2*np.pi, len(nearby_uavs))  # Random phases
        amplitudes, phases = self.quantum_position_encoding(uav, nearby_uavs)
        
        # Update internal quantum state with decay
        if self.quantum_state is None:
            self.quantum_state = amplitudes
        interference_force = np.zeros(2)
        
        if len(nearby_uavs) < 2:
            return interference_force
            
        # Ensure quantum state and phase angles are initialized for this number of UAVs
        # This fixes the index out of range error and improves stability
        if self.quantum_state is None or len(self.quantum_state) < len(nearby_uavs):
            # Initialize quantum state with equal probability distribution
            self.quantum_state = np.ones(len(nearby_uavs)) / len(nearby_uavs)
            # Initialize phase angles randomly between 0 and 2π
            self.phase_angles = np.random.uniform(0, 2*np.pi, len(nearby_uavs))
        
        # For each pair of UAVs, calculate interference effects
        for i in range(len(nearby_uavs)):
            for j in range(i+1, len(nearby_uavs)):
                # Get positions
                pos_i = nearby_uavs[i].position
                pos_j = nearby_uavs[j].position
                
                # Calculate midpoint between UAVs
                midpoint = (pos_i + pos_j) / 2.0
                
                # Calculate interference amplitude using quantum probability
                amp_i = self.quantum_state[i] 
                amp_j = self.quantum_state[j]
                phase_i = self.phase_angles[i]
                phase_j = self.phase_angles[j]
                
                # Convert to complex numbers for interference calculation
                psi_i = cmath.rect(np.sqrt(amp_i), phase_i)
                psi_j = cmath.rect(np.sqrt(amp_j), phase_j)
                
                # Calculate interference term (constructive or destructive)
                interference = 2 * np.sqrt(amp_i * amp_j) * np.cos(phase_i - phase_j)
                
                # Direction from UAV to interference point
                direction = midpoint - uav.position
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    # Force is stronger when constructive interference (>0), weaker when destructive (<0)
                    force_magnitude = self.interference_factor * interference * CONFIG["uav_max_acceleration"]
                    force = direction / distance * force_magnitude
                    interference_force += force
        
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(interference_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            interference_force = interference_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return interference_force
        
    def calculate_entanglement_force(self, uav, target_uav, entangled_uavs):
        """
        Calculate force based on quantum entanglement between UAVs
        
        Args:
            uav: The UAV to calculate force for
            target_uav: The target UAV entangled with this UAV
            entangled_uavs: List of all entangled UAVs in system
            
        Returns:
            numpy.ndarray: The entanglement force vector [x, y]
        """
        if target_uav is None:
            return np.zeros(2)
            
        # Extract relative position
        relative_pos = target_uav.position - uav.position
        distance = np.linalg.norm(relative_pos)
        
        if distance < 1e-10:
            return np.zeros(2)
            
        # Calculate entanglement strength based on distance
        # Quantum entanglement is not affected by distance classically, but we model
        # a distance-based effect for practical purposes
        entanglement_factor = self.entanglement_strength * (1.0 / (1.0 + 0.1 * distance))
        
        # Calculate the difference in velocities
        velocity_diff = target_uav.velocity - uav.velocity
        
        # Calculate entanglement force - tends to synchronize velocities
        # of entangled UAVs in a non-local way
        entanglement_force = velocity_diff * entanglement_factor
        
        # Add quantum tunneling effect - allows "jumping" toward entangled UAV
        tunneling_probability = np.exp(-distance / self.quantum_sigma)
        tunneling_force = relative_pos / distance * tunneling_probability * self.entanglement_strength
        
        # Combine forces
        combined_force = entanglement_force + tunneling_force
        
        # Limit force to max acceleration
        force_magnitude = np.linalg.norm(combined_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            combined_force = combined_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return combined_force
        
    def quantum_probability_field(self, uav, environment_grid, nearby_uavs):
        """
        Calculate quantum probability field for optimal positioning
        
        Args:
            uav: The UAV to calculate field for
            environment_grid: Grid of environment states
            nearby_uavs: List of nearby UAVs
            
        Returns:
            numpy.ndarray: Grid of probability amplitudes for optimal positions
        """
        if environment_grid is None or not nearby_uavs:
            return None
            
        # Create probability amplitude field
        grid_shape = environment_grid.shape
        probability_field = np.zeros(grid_shape)
        
        # Environment features (obstacles, etc.)
        # Higher values in environment grid represent less desirable positions
        
        # For each grid cell, calculate quantum amplitude
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                # Convert grid coordinates to world coordinates
                grid_pos = np.array([i, j]) * CONFIG["scenario_width"] / grid_shape[0]
                
                # Calculate wave function amplitude for this position
                amplitude = 0.0
                
                # Contribution from UAV's current position (Gaussian)
                distance = np.linalg.norm(grid_pos - uav.position)
                self_contribution = np.exp(-distance**2 / self.quantum_sigma**2)
                
                # Contributions from nearby UAVs (destructive interference)
                neighbor_contribution = 0.0
                for other_uav in nearby_uavs:
                    if other_uav is uav:
                        continue
                        
                    neighbor_distance = np.linalg.norm(grid_pos - other_uav.position)
                    phase_diff = np.sin(neighbor_distance * np.pi * 2 / self.quantum_sigma)
                    neighbor_contribution += np.exp(-neighbor_distance**2 / (2 * self.quantum_sigma**2)) * phase_diff
                
                # Environment contribution (obstacles decrease amplitude)
                environment_contribution = 1.0 - environment_grid[i, j]
                
                # Combine contributions with interference
                amplitude = self_contribution + self.interference_factor * neighbor_contribution
                amplitude *= environment_contribution
                
                # Store in probability field
                probability_field[i, j] = max(0.0, amplitude)
                
        # Normalize
        total = np.sum(probability_field) + 1e-10
        probability_field /= total
        
        return probability_field
        
    def find_optimal_position(self, uav, probability_field, grid_shape):
        """
        Find optimal position using quantum probability field
        
        Args:
            uav: The UAV to calculate for
            probability_field: Quantum probability field
            grid_shape: Shape of the environment grid
            
        Returns:
            numpy.ndarray: Optimal position vector [x, y]
        """
        if probability_field is None:
            return None
            
        # Find position with highest probability
        max_idx = np.unravel_index(np.argmax(probability_field), probability_field.shape)
        
        # Convert grid index to world coordinates
        optimal_position = np.array([max_idx[0], max_idx[1]]) * CONFIG["scenario_width"] / grid_shape[0]
        
        return optimal_position
        
    def calculate_quantum_force(self, uav, nearby_uavs, environment_grid=None, step_count=None):
        """
        Calculate overall quantum force for UAV
        
        Args:
            uav: The UAV to calculate force for
            nearby_uavs: List of nearby UAVs
            environment_grid: Optional environment grid
            step_count: Current simulation step for more precise cache control
            
        Returns:
            numpy.ndarray: The quantum force vector [x, y]
        """
        # EXTREME PERFORMANCE MODE: Skip ALL quantum calculations in performance mode
        # This maintains basic functionality while dramatically improving performance
        if self.performance_mode and hasattr(uav, 'id'):
            # If in aggressive performance mode, only compute every N steps
            if step_count is not None:
                should_calculate = (step_count % self.calculation_interval == 0)
            else:
                # Default to time-based calculation if step count not provided
                current_time = int(time.time() * 5)  # Get time in 200ms increments
                should_calculate = (not hasattr(self, 'last_quantum_update') or 
                                  current_time != self.last_quantum_update)
                
                if should_calculate:
                    self.last_quantum_update = current_time
            
            # Check cache for this UAV id
            cache_key = str(uav.id)
            if cache_key in self.cached_results:
                cache_entry = self.cached_results[cache_key]
                if not should_calculate and time.time() - cache_entry['time'] < self.cache_lifetime:
                    # Use cached value if it's still valid
                    return cache_entry['force']
            
            # If we need to calculate (cache expired or first time)
            if should_calculate:
                # Apply quantum interference effects - with full computation
                interference_force = self.apply_quantum_interference(uav, nearby_uavs)
                
                # Store in cache with timestamp
                self.cached_results[cache_key] = {
                    'force': interference_force,
                    'time': time.time()
                }
                return interference_force
            elif cache_key in self.cached_results:
                # Return most recent cached result even if we should calculate but haven't yet
                return self.cached_results[cache_key]['force']
            
        # Fall back to standard calculation if performance mode disabled
        # or if we don't have a valid UAV id for caching
        interference_force = self.apply_quantum_interference(uav, nearby_uavs)
        
        # Find entangled partner (closest UAV)
        entangled_partner = None
        min_distance = float('inf')
        
        for other_uav in nearby_uavs:
            if other_uav is uav:
                continue
                
            distance = np.linalg.norm(uav.position - other_uav.position)
            if distance < min_distance:
                min_distance = distance
                entangled_partner = other_uav
                
        # Calculate entanglement force (skip in performance mode)
        if self.performance_mode and step_count is not None and step_count % 3 != 0:
            # In performance mode, approximate entanglement force
            if entangled_partner is not None:
                direction = entangled_partner.position - uav.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    entanglement_force = direction / distance * self.entanglement_strength
                else:
                    entanglement_force = np.zeros(2)
            else:
                entanglement_force = np.zeros(2)
        else:
            # Full calculation
            entanglement_force = self.calculate_entanglement_force(uav, entangled_partner, nearby_uavs)
        
        # Calculate quantum probability field if environment grid provided
        optimal_force = np.zeros(2)
        if environment_grid is not None:
            probability_field = self.quantum_probability_field(uav, environment_grid, nearby_uavs)
            optimal_position = self.find_optimal_position(uav, probability_field, environment_grid.shape)
            
            if optimal_position is not None:
                direction = optimal_position - uav.position
                distance = np.linalg.norm(direction)
                
                if distance > 0:
                    optimal_force = direction / distance * min(distance, CONFIG["uav_max_acceleration"])
        
        # Combine quantum forces
        quantum_force = (
            interference_force * 0.4 +
            entanglement_force * 0.4 +
            optimal_force * 0.2
        )
        
        # Limit to max acceleration
        force_magnitude = np.linalg.norm(quantum_force)
        if force_magnitude > CONFIG["uav_max_acceleration"]:
            quantum_force = quantum_force / force_magnitude * CONFIG["uav_max_acceleration"]
            
        return quantum_force
