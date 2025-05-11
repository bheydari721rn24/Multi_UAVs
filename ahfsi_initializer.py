# AHFSI Framework Initialization and Patch Module
# This module ensures that AHFSI components work properly

import os
import sys
import importlib
import importlib.util


# Function to check if a module can be imported
def module_exists(module_name):
    return importlib.util.find_spec(module_name) is not None

# Define required dependencies
REQUIRED_MODULES = {
    'scipy.stats': 'scipy_mock',  # Use our mock if scipy.stats is not available
}

# Patch system modules if needed
def patch_modules():
    patched = []
    
    for module_name, fallback in REQUIRED_MODULES.items():
        if not module_exists(module_name):
            # If the module doesn't exist, add a fallback
            parts = module_name.split('.')
            parent_module = '.'.join(parts[:-1]) if len(parts) > 1 else None
            fallback_path = os.path.join(os.path.dirname(__file__), f'{fallback}.py')
            
            if os.path.exists(fallback_path):
                # Import the fallback module
                fallback_module = importlib.import_module(fallback)
                
                # Add to sys.modules to simulate the missing module
                sys.modules[module_name] = fallback_module
                patched.append(module_name)
                print(f"[AHFSI Initializer] Patched {module_name} with {fallback}")
    
    return patched

# Initialize the AHFSI framework by patching modules
def initialize():
    patched_modules = patch_modules()
    return len(patched_modules) > 0

# Run initialization when this module is imported
initialization_result = initialize()
