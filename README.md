# Multi-UAV Roundup Strategy with CEL-MADDPG

This project implements the "Multi-UAV roundup strategy method based on deep reinforcement learning CEL-MADDPG algorithm" as described in the paper published in Expert Systems With Applications. The implementation includes:

1. **Curriculum Experience Learning (CEL)**: Divides the roundup task into three subtasks
2. **Preferential Experience Replay (PER)**: Selects more important samples for learning
3. **Relative Experience Learning (REL)**: Uses experiences similar to the current situation

## Project Structure

- `run_simulation.py`: Main script to run training, testing or demo
- `environment.py`: Multi-UAV roundup environment simulation
- `agents.py`: UAV and target agent implementations
- `cel_maddpg.py`: Implementation of the CEL-MADDPG algorithm
- `networks.py`: Neural network models for actor and critic
- `replay_buffer.py`: Experience replay buffer with PER and REL strategies
- `visualization.py`: Visual display and animation with military-radar style
- `utils.py`: Utility functions and configurations

## Installation

### Requirements

To run this project, you need:

- Python 3.8 or higher
- TensorFlow 2.9 or higher
- NumPy
- Matplotlib
- Pillow
- FFmpeg (optional, for high-quality video output)

### Installing Dependencies

Place the `requirements.txt` file in your project folder and install the dependencies with:

```bash
pip install -r requirements.txt# Multi-UAV roundup strategy method based on deep reinforcement learning CEL-MADDPG algorithm
