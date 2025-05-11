# Military Visualization Selector for Multi-UAV System
# Provides an interface to use military-style radar visualization

# Import visualization system
from visualization import Visualizer as MilitaryVisualizer
from environment import Environment
from utils import CONFIG


def create_visualizer(env, **kwargs):
    """
    Create a military-style visualizer
    
    Args:
        env: Environment object
        **kwargs: Additional parameters (ignored)
        
    Returns:
        Military visualizer object
    """
    return MilitaryVisualizer(env)


def run_demo(enable_ahfsi=True, num_uavs=5, num_obstacles=8):
    """
    Run a demonstration of the military visualization system
    
    Args:
        enable_ahfsi: Whether to enable AHFSI in the environment
        num_uavs: Number of UAVs to simulate
        num_obstacles: Number of obstacles to place
    """
    # Create environment
    env = Environment(num_uavs=num_uavs, num_obstacles=num_obstacles, enable_ahfsi=enable_ahfsi)
    env.reset()
    
    print("\nRunning Military-Style Visualization Demo")
    print("-" * 50)
    military_visualizer = create_visualizer(env)
    
    # Run a few steps to show movement
    for step in range(20):
        # Generate random actions in the correct format
        # The environment expects actions as numpy arrays of shape (2,)
        import numpy as np
        actions = []
        for _ in range(len(env.uavs)):
            actions.append(np.array([0.1, 0.1]))  # Correct format for environment
        
        # Step environment
        env.step(actions)
        
        # Render
        military_visualizer.render(episode=0, step=step)


if __name__ == "__main__":
    # Parse command line arguments if provided
    import argparse
    parser = argparse.ArgumentParser(description="Military Tactical Visualization Demo")
    parser.add_argument("--ahfsi", type=bool, default=True,
                        help="Enable AHFSI framework")
    parser.add_argument("--uavs", type=int, default=5,
                        help="Number of UAVs")
    parser.add_argument("--obstacles", type=int, default=8,
                        help="Number of obstacles")
    
    args = parser.parse_args()
    
    # Run demo with parsed arguments
    run_demo(
        enable_ahfsi=args.ahfsi,
        num_uavs=args.uavs,
        num_obstacles=args.obstacles
    )
