# AHFSI Framework Manager
# This is a central manager for AHFSI components that ensures
# consistent behavior across the entire application

import os
import sys
import importlib
import numpy as np

# Global status flags
AHFSI_AVAILABLE = False
AHFSI_COMPONENTS_AVAILABLE = False

# Global registry of implementations
_implementations = {}

# Flag to track initialization
_initialized = False

def _register_implementation(name, implementation):
    """Register an implementation in the global registry"""
    global _implementations
    _implementations[name] = implementation

def get_implementation(name):
    """Get an implementation by name, returning None if not available"""
    global _implementations
    return _implementations.get(name)

def initialize():
    """Initialize the AHFSI framework and register all components"""
    global AHFSI_AVAILABLE, AHFSI_COMPONENTS_AVAILABLE, _initialized
    
    # Only initialize once
    if _initialized:
        return (AHFSI_AVAILABLE, AHFSI_COMPONENTS_AVAILABLE)
    
    # Create base implementations
    
    # Basic AHFSIController
    class DummyAHFSIController:
        """Dummy implementation of AHFSIController that maintains API compatibility"""
        def __init__(self, *args, **kwargs):
            self.uav_id = kwargs.get('uav_id', 0) if kwargs else (args[0] if args else 0)
            self.role = "PURSUER"  # Default role
            self.swarm_weight = 0.0  # No swarm behavior by default
            
        def process_rl_action(self, uav, nearby_uavs, rl_action, env_state):
            """Pass through RL action unchanged"""
            return rl_action
            
        def get_augmented_state(self, state):
            """Return state unchanged"""
            return state
            
        def get_state_dim_extension(self):
            """No state extension"""
            return 0
    
    # Basic SwarmBehavior
    class DummySwarmBehavior:
        """Dummy implementation of SwarmBehavior that maintains API compatibility"""
        def __init__(self):
            """Initialize with default config"""
            self.config = {"behaviors": {}}
            
        def calculate_cohesion_force(self, uav, nearby_uavs):
            """Return zero force"""
            return np.zeros(2)
            
        def calculate_separation_force(self, uav, nearby_uavs):
            """Return zero force"""
            return np.zeros(2)
            
        def calculate_alignment_force(self, uav, nearby_uavs):
            """Return zero force"""
            return np.zeros(2)
            
        def calculate_formation_force(self, uav, nearby_uavs, target_position=None, formation_type=None):
            """Return zero force"""
            return np.zeros(2)
    
    # Basic InformationTheory
    class DummyInformationTheory:
        """Dummy implementation of InformationTheory that maintains API compatibility"""
        def __init__(self, *args, **kwargs):
            """Initialize with default parameters"""
            self.config = {}
            
        def calculate_information_force(self, uav, nearby_uavs, target_position=None):
            """Return zero force"""
            return np.zeros(2)
            
        def calculate_coverage_metric(self, uav_positions):
            """Return neutral coverage metric"""
            return 0.5
    
    # Register dummy implementations
    _register_implementation('AHFSIController', DummyAHFSIController)
    _register_implementation('SwarmBehavior', DummySwarmBehavior)
    _register_implementation('InformationTheory', DummyInformationTheory)
    
    # Try first to use the fixed implementation if available
    try:
        # Import from our simplified fixed implementation first
        ahfsi_fixed = importlib.import_module('ahfsi_fixed')
        
        # Register the components from the fixed implementation
        _register_implementation('AHFSIController', ahfsi_fixed.AHFSIController)
        _register_implementation('SwarmBehavior', ahfsi_fixed.SwarmBehavior)
        _register_implementation('InformationTheory', ahfsi_fixed.InformationTheory)
        
        # Set status flags based on fixed implementation
        AHFSI_AVAILABLE = ahfsi_fixed.AHFSI_AVAILABLE
        AHFSI_COMPONENTS_AVAILABLE = ahfsi_fixed.AHFSI_COMPONENTS_AVAILABLE
        
        # Don't print anything here as the fixed module already prints a message
        
    except ImportError:
        # If fixed implementation is not available, try original modules
        try:
            # Try to import the real AHFSI modules
            real_controller = importlib.import_module('ahfsi_framework').AHFSIController
            _register_implementation('AHFSIController', real_controller)
            AHFSI_AVAILABLE = True
            
            # Try to import component modules
            real_swarm = importlib.import_module('swarm_intelligence').SwarmBehavior
            real_info = importlib.import_module('information_theory').InformationTheory
            
            _register_implementation('SwarmBehavior', real_swarm)
            _register_implementation('InformationTheory', real_info)
            AHFSI_COMPONENTS_AVAILABLE = True
            
            print("AHFSI framework and components loaded successfully")
            
        except ImportError:
            # If all imports fail, we'll just use the dummy implementations we registered earlier
            print("Using AHFSI compatible substitutes - original framework not available")
    
    # Mark as initialized
    _initialized = True
    
    return (AHFSI_AVAILABLE, AHFSI_COMPONENTS_AVAILABLE)

# Run initialization when module is imported
initialize()

# Explicitly export the implementations
AHFSIController = get_implementation('AHFSIController')
SwarmBehavior = get_implementation('SwarmBehavior')
InformationTheory = get_implementation('InformationTheory')
