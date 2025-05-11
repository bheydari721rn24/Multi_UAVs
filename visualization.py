# Visualization module for Multi-UAV simulation environment
# Provides real-time graphical representation of UAVs, target, and obstacles
# Enhanced with better obstacle visibility (10x larger display) and removed UAV trajectories

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import matplotlib.transforms as transforms
import os
from utils import CONFIG # Import shared configuration parameters
import matplotlib.patheffects as path_effects
import time


class Visualizer:
    """Visualization system for the Multi-UAV simulation environment.
    
    Handles real-time rendering of UAVs, target, obstacles, and sensor data.
    Features a radar-like military style visualization with specialized graphics
    for monitoring tactical scenarios and obstacle avoidance behaviors.
    
    Key visualization features:
    - Obstacles are rendered 10x larger than physical size for better visibility
    - UAV trajectory paths (black lines) are disabled for cleaner visuals
    - Red polygon shows target encirclement by UAVs
    """
    def __init__(self, env):
        """Initialize the visualization system.
        
        Args:
            env: Reference to the simulation environment
        """
        self.env = env  # Store reference to environment
        
        # Use dark theme for better contrast and military-style visuals
        plt.style.use('dark_background')
        
        # Create figure with dark green background (radar-like military aesthetic)
        self.fig, self.ax = plt.subplots(figsize=(16, 16), dpi=150, facecolor='#0a3b0a')
        self.ax.set_facecolor('#093D09')  # Slightly lighter green for plot area
        
        # Title with radar-like styling (green text with stroke effect)
        title = self.fig.suptitle('Multi-UAV Military Tactical Simulation', fontsize=20, color='#00ff00', y=0.98, fontweight='bold')
        title.set_path_effects([path_effects.withStroke(linewidth=2, foreground='#005500')])
        
        # Get scenario dimensions from global configuration
        self.width = CONFIG["scenario_width"]    # Width in km (15 km)
        self.height = CONFIG["scenario_height"]  # Height in km (15 km)
        
        # Setup for smoother real-time animation and rendering
        self.fig.canvas.draw()
        plt.ion()  # Enable interactive mode for real-time updates
        
        # Animation configuration
        self.stored_frames = []  # For saving video if requested
        self.frame_interval = 0.01  # 100 FPS target for smooth animation
        
        # Initialize visualization elements (will be populated during setup)
        self.uav_patches = []             # UAV triangle markers
        self.uav_capture_circles = []     # Circles showing UAV capture range
        self.uav_trajectory_plots = []    # UAV path history (disabled per user request)
        
        self.target_patch = None          # Target marker
        self.target_trajectory_plot = None # Target path history
        
        # Obstacle visualization elements
        self.obstacle_plots = []          # Main obstacle circles (10x physical size)
        self.obstacle_radar_rings = []    # Animated radar rings around obstacles
        self.obstacle_scan_angles = []    # Rotating scan lines for obstacles
        
        # Task state text display with radar-like styling
        self.task_state_text = self.ax.text(
            0.02, 0.98, "",
            transform=self.ax.transAxes,
            fontsize=16,
            color='#00ff00',  # Bright green radar text
            verticalalignment='top',
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#003300', alpha=0.85, pad=0.8, edgecolor='#00aa00'),
            path_effects=[path_effects.withStroke(linewidth=1.5, foreground='#005500')]
        )
        
        # Encirclement visualization
        self.encirclement_polygon = None
        # Polygon for UAV encirclement hull
        self.hull_poly = None
        
        # Set up the plot
        self._setup_plot()
        
    def _setup_plot(self):
        """Set up the initial plot configuration with radar-like styling."""
        # Set plot limits
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        
        # Set labels with larger font
        self.ax.set_xlabel('X Position (km)', fontsize=18, color='white')
        self.ax.set_ylabel('Y Position (km)', fontsize=18, color='white')
        
        # Tick settings - increased for readability
        self.ax.tick_params(colors='white', labelsize=16)
        
        # Restore ORIGINAL radar-like background
        # Create dense military-style grid with many more lines in all directions
        # Turn off default grid to use our own completely customized grid
        self.ax.grid(False)
        
        # Define multiple levels of grid density for a more realistic military display
        grid_spacing_major = 1.0    # Major grid lines every 1 unit
        grid_spacing_minor = 0.25   # Minor grid lines every 0.25 units
        grid_spacing_micro = 0.1    # Micro grid lines every 0.1 units
        
        # Major grid lines (vertical) - most visible
        for x in np.arange(0, self.width+0.1, grid_spacing_major):
            self.ax.axvline(x=x, color='#00ff00', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Major grid lines (horizontal) - most visible
        for y in np.arange(0, self.height+0.1, grid_spacing_major):
            self.ax.axhline(y=y, color='#00ff00', linestyle='-', alpha=0.3, linewidth=0.8)
            
        # Minor grid lines (vertical) - medium visibility
        for x in np.arange(0, self.width+0.1, grid_spacing_minor):
            if x % grid_spacing_major != 0:  # Skip where major lines already exist
                self.ax.axvline(x=x, color='#00ff00', linestyle='-', alpha=0.15, linewidth=0.5)
        
        # Minor grid lines (horizontal) - medium visibility
        for y in np.arange(0, self.height+0.1, grid_spacing_minor):
            if y % grid_spacing_major != 0:  # Skip where major lines already exist
                self.ax.axhline(y=y, color='#00ff00', linestyle='-', alpha=0.15, linewidth=0.5)
        
        # Micro grid lines (vertical) - barely visible for dense background texture
        for x in np.arange(0, self.width+0.1, grid_spacing_micro):
            if x % grid_spacing_minor != 0:  # Skip where minor/major lines already exist
                self.ax.axvline(x=x, color='#00ff00', linestyle='-', alpha=0.05, linewidth=0.3)
        
        # Micro grid lines (horizontal) - barely visible for dense background texture
        for y in np.arange(0, self.height+0.1, grid_spacing_micro):
            if y % grid_spacing_minor != 0:  # Skip where minor/major lines already exist
                self.ax.axhline(y=y, color='#00ff00', linestyle='-', alpha=0.05, linewidth=0.3)
                
        # Add diagonal grid lines for more complex tactical display
        # Diagonal lines from bottom-left to top-right (45 degrees)
        for offset in np.arange(-self.width, self.width+self.height, grid_spacing_major*2):
            self.ax.plot([max(0, offset), min(self.width, offset+self.height)],
                         [max(0, -offset), min(self.height, self.width-offset)],
                         color='#00ff00', linestyle='-', alpha=0.1, linewidth=0.4)
                
        # Diagonal lines from bottom-right to top-left (135 degrees)
        for offset in np.arange(0, self.width+self.height, grid_spacing_major*2):
            self.ax.plot([max(0, self.width-offset), min(self.width, 2*self.width-offset)],
                         [max(0, offset-self.width), min(self.height, offset)],
                         color='#00ff00', linestyle='-', alpha=0.1, linewidth=0.4)
        
        # Create radar sweep effect (concentric circles from center)
        center_x, center_y = self.width / 2, self.height / 2
        for radius in range(5, int(max(self.width, self.height)), 10):
            circle = patches.Circle(
                (center_x, center_y), radius,
                fill=False, 
                edgecolor='#00ff00', 
                linestyle='-', 
                linewidth=0.8, 
                alpha=0.3, 
                zorder=1
            )
            self.ax.add_patch(circle)
        
        # Create radial lines from center
        for angle in range(0, 360, 30):
            rad_angle = np.radians(angle)
            dx = np.cos(rad_angle) * max(self.width, self.height)
            dy = np.sin(rad_angle) * max(self.width, self.height)
            self.ax.plot(
                [center_x, center_x + dx], 
                [center_y, center_y + dy], 
                color='#00ff00', 
                linewidth=0.5, 
                alpha=0.2, 
                zorder=1
            )
            
        # Add circular radar sweep lines
        center_x, center_y = self.width / 2, self.height / 2
        for radius in range(5, int(max(self.width, self.height)), 10):
            circle = patches.Circle((center_x, center_y), radius, 
                                   fill=False, edgecolor='#00ff00', linestyle='-', 
                                   linewidth=0.8, alpha=0.5, zorder=1)
            self.ax.add_patch(circle)
        
        # Add more prominent circular tactical range indicators
        center_x, center_y = self.width/2, self.height/2
        # Calculate the radius needed to reach the corners
        max_radius = np.sqrt((self.width**2 + self.height**2)) / 2
        
        # Create more professional-looking range circles with better spacing and visibility
        for r in np.linspace(0.2, max_radius, 8):
            # Add subtle pulsing effect to circles by varying alpha
            alpha = 0.1 + 0.05 * np.sin(r * 3)  # Subtle pulsing based on radius
            circle = plt.Circle(
                (center_x, center_y), r, fill=False, 
                color='#00ff00', linestyle='-', alpha=alpha,
                path_effects=[path_effects.withSimplePatchShadow(offset=(0, 0), shadow_rgbFace='#00ff00', alpha=0.15)]
            )
            self.ax.add_patch(circle)
        
        # Add more radial lines for better tactical display effect - with slight gradient effect
        for i, angle in enumerate(np.linspace(0, 2*np.pi, 16, endpoint=False)):
            # Alternate line styles and colors for visual interest
            if i % 4 == 0:  # Major radial lines
                linestyle, alpha, color = '-', 0.25, '#00ff00'
            else:  # Minor radial lines
                linestyle, alpha, color = '--', 0.15, '#00aa00'
                
            dx, dy = max_radius * np.cos(angle), max_radius * np.sin(angle)
            self.ax.plot([center_x, center_x + dx], [center_y, center_y + dy], 
                         color=color, linestyle=linestyle, alpha=alpha)
        
        # Draw boundary with enhanced military styling
        # Outer glow effect
        outer_boundary = patches.Rectangle(
            (-0.1, -0.1), self.width + 0.2, self.height + 0.2,
            linewidth=2.5, edgecolor='#00ff00', facecolor='none', linestyle='-', alpha=0.3,
            path_effects=[path_effects.withStroke(linewidth=4, foreground='#003300')]
        )
        self.ax.add_patch(outer_boundary)
        
        # Main boundary with military styling
        boundary = patches.Rectangle(
            (0, 0), self.width, self.height,
            linewidth=3.5, edgecolor='#00ff00', facecolor='none', linestyle='-',
            path_effects=[path_effects.withStroke(linewidth=4.5, foreground='#003300')]
        )
        self.ax.add_patch(boundary)
        
        # Store obstacle animation elements
        self.obstacle_radar_rings = []
        self.obstacle_scan_angles = []
        
        # Draw obstacles with proper military radar scanning animation
        for obstacle in self.env.obstacles:
            # Main obstacle base - dark blue as requested
            obstacle_plot = patches.Circle(
                (obstacle.position[0], obstacle.position[1]), 
                obstacle.radius * 10,  # Enlarged for better visibility
                linewidth=2.5, 
                edgecolor='#0055AA', 
                facecolor='#003388', 
                alpha=0.9,
                zorder=2,
                path_effects=[path_effects.withStroke(linewidth=3.5, foreground='#001155')]
            )
            
            # Initialize radar scan angle for animation
            scan_angle = np.random.uniform(0, 2*np.pi)  # Random starting angle
            self.obstacle_scan_angles.append(scan_angle)
            
            # Create radar lines INSIDE the obstacle circle - make sure they stay inside
            # Use the same radius as the blue circle patch for perfect alignment
            visual_radius = obstacle_plot.get_radius()
            scan_length = visual_radius  # Use the exact same visual radius
            
            # Calculate endpoint coordinates based on angle
            end_x = obstacle.position[0] + scan_length * np.cos(scan_angle)
            end_y = obstacle.position[1] + scan_length * np.sin(scan_angle)
            
            # Create the red scan line fully inside the obstacle
            scan_line, = self.ax.plot(
                [obstacle.position[0], end_x],
                [obstacle.position[1], end_y],
                color='#FF0000',  # Red line as requested
                linewidth=2.0,
                solid_capstyle='round',
                alpha=0.9,
                zorder=3
            )
            
            self.ax.add_patch(obstacle_plot)
            self.obstacle_plots.append(obstacle_plot)
            self.obstacle_radar_rings.append(scan_line)
        
        # Create objects for UAVs - ALL BLACK as requested
        uav_color = 'black'  # All UAVs are black
        for i, uav in enumerate(self.env.uavs):
            # UAV capture circle
            capture_circle = patches.Circle(
                (0, 0), CONFIG["capture_distance"],
                linewidth=3.5, edgecolor=uav_color,
                facecolor=uav_color, alpha=0.25,
                linestyle='--'
            )
            self.ax.add_patch(capture_circle)
            self.uav_capture_circles.append(capture_circle)
            
            # Intentionally disabled UAV trajectory plots (black paths of UAVs)
            # Create empty plot objects to maintain code structure but not show actual trajectories
            # This provides cleaner visualization by hiding the path history of UAVs
            trajectory_plot, = self.ax.plot(
                [], [], '-', linewidth=0, color='none',
                alpha=0.0
            )
            self.uav_trajectory_plots.append(trajectory_plot)
            
            # Create military-style UAV representation
            drone_parts = self._create_drone(uav.position, uav.yaw, uav_color)
            # Add each drone part to the plot
            for part in drone_parts:
                if isinstance(part, patches.Circle) or isinstance(part, patches.Polygon):
                    self.ax.add_patch(part)
            # Store the collection of parts
            self.uav_patches.append(drone_parts)
        
        # Create target as a visually distinct circle patch
        target_patch = patches.Circle(
            (self.env.target.position[0], self.env.target.position[1]),
            radius=0.15,  # Circle radius
            facecolor='#FF0000',  # Bright red color
            edgecolor='white',
            linewidth=3.0,
            zorder=10,  # Ensure it's on top
            path_effects=[path_effects.withStroke(linewidth=4, foreground='#004080')]
        )
        self.ax.add_patch(target_patch)
        self.target_patch = target_patch
        
        # Target trajectory with enhanced glowing effect
        self.target_trajectory_plot, = self.ax.plot(
            [], [], '-', linewidth=3, color='#0080FF',
            alpha=0.8, path_effects=[path_effects.withSimplePatchShadow(
                offset=(0, 0), shadow_rgbFace='#0080FF', alpha=0.4)]
        )
        
        # Add a legend with radar-like styling
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor='black', edgecolor='white', label='UAVs', linewidth=1.5),
            Patch(facecolor='#0080FF', edgecolor='white', label='Target', linewidth=1.5),
            Line2D([0], [0], linestyle='--', color='black', lw=2.5, label='Capture Range'),
            Patch(facecolor='#DD4400', edgecolor='#FF5500', label='Obstacle & Radar', linewidth=1.5)
        ]
        
        # Add to legend with larger font
        legend = self.ax.legend(
            handles=legend_elements, 
            loc='upper right',
            fontsize=18, 
            framealpha=0.7,
            facecolor='black', 
            edgecolor='#00ff00'
        )
        
        # Style the legend background and border
        legend.get_frame().set_facecolor('#003300')
        legend.get_frame().set_edgecolor('#00ff00')
        legend.get_frame().set_linewidth(2)
        
        # Style the text elements
        for text in legend.get_texts():
            text.set_color('white')
            
    def _create_drone(self, position, angle, color):
        """Create a military-style UAV representation."""
        # Base size for the military drone
        size = 0.25  # Slightly larger for better visibility
        
        # Create a group of patches for the drone
        drone_parts = []
        
        # Military drone shape - elongated body
        # Main body shape - elongated hexagon
        body_length = size * 1.6
        body_width = size * 0.6
        
        # Create body vertices for a military drone shape
        body_vertices = np.array([
            [body_length/2, 0],                    # Nose
            [body_length/4, body_width/2],         # Right shoulder
            [-body_length/2, body_width/3],        # Right tail
            [-body_length/2, -body_width/3],       # Left tail
            [body_length/4, -body_width/2],        # Left shoulder
        ])
        
        # Rotate vertices
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotated_vertices = np.dot(body_vertices, rotation_matrix.T)
        
        # Translate to position
        translated_vertices = rotated_vertices + position
        
        # Create military drone body
        body = patches.Polygon(
            translated_vertices,
            closed=True,
            facecolor=color,
            edgecolor='#FFFFFF',
            linewidth=2.0,
            zorder=10,
            path_effects=[path_effects.withStroke(linewidth=3.0, foreground='#111111')]
        )
        drone_parts.append(body)
        
        # Add canopy/cockpit
        canopy_center_x = position[0] + np.cos(angle) * body_length/6
        canopy_center_y = position[1] + np.sin(angle) * body_length/6
        canopy = patches.Ellipse(
            (canopy_center_x, canopy_center_y),
            body_length/3.5,
            body_width/2,
            angle=np.degrees(angle),
            facecolor='#AADDFF',
            edgecolor='#FFFFFF',
            linewidth=1.0,
            alpha=0.7,
            zorder=11
        )
        drone_parts.append(canopy)
        
        # Add wings
        wing_width = body_length * 0.8
        wing_length = body_width * 0.7
        
        # Left and right wing vertices
        left_wing_vertices = np.array([
            [0, 0],                            # Wing root front
            [-wing_length/2, -wing_width/2],   # Wing tip front
            [-wing_length, -wing_width],       # Wing tip
            [-wing_length/2, -wing_width/1.5], # Wing tip back
            [0, -wing_width/4]                 # Wing root back
        ])
        
        right_wing_vertices = np.array([
            [0, 0],                            # Wing root front
            [-wing_length/2, wing_width/2],    # Wing tip front
            [-wing_length, wing_width],        # Wing tip
            [-wing_length/2, wing_width/1.5],  # Wing tip back
            [0, wing_width/4]                  # Wing root back
        ])
        
        # Wing position offset from center
        wing_offset_x = -size/4
        
        # Position of wing attachment
        wing_pos_x = position[0] + np.cos(angle) * wing_offset_x
        wing_pos_y = position[1] + np.sin(angle) * wing_offset_x
        wing_pos = np.array([wing_pos_x, wing_pos_y])
        
        # Rotate wing vertices
        left_wing_rotated = np.dot(left_wing_vertices, rotation_matrix.T)
        right_wing_rotated = np.dot(right_wing_vertices, rotation_matrix.T)
        
        # Translate wings to position
        left_wing_translated = left_wing_rotated + wing_pos
        right_wing_translated = right_wing_rotated + wing_pos
        
        # Create wing patches
        left_wing = patches.Polygon(
            left_wing_translated,
            closed=True,
            facecolor=color,
            edgecolor='#DDDDDD',
            linewidth=1.5,
            alpha=0.9,
            zorder=9
        )
        
        right_wing = patches.Polygon(
            right_wing_translated,
            closed=True,
            facecolor=color,
            edgecolor='#DDDDDD',
            linewidth=1.5,
            alpha=0.9,
            zorder=9
        )
        
        drone_parts.append(left_wing)
        drone_parts.append(right_wing)
        
        # Add tail fins
        tail_size = size * 0.4
        tail_pos_x = position[0] - np.cos(angle) * body_length/2
        tail_pos_y = position[1] - np.sin(angle) * body_length/2
        
        # Vertical tail fin
        tail_fin_vertices = np.array([
            [0, 0],                    # Tail fin root
            [-tail_size/2, tail_size], # Tail fin tip
            [-tail_size, 0]            # Tail fin back
        ])
        
        # Rotate tail fin
        tail_fin_rotated = np.dot(tail_fin_vertices, rotation_matrix.T)
        
        # Translate tail fin
        tail_fin_translated = tail_fin_rotated + np.array([tail_pos_x, tail_pos_y])
        
        # Create tail fin patch
        tail_fin = patches.Polygon(
            tail_fin_translated,
            closed=True,
            facecolor=color,
            edgecolor='#DDDDDD',
            linewidth=1.0,
            alpha=0.9,
            zorder=9
        )
        
        drone_parts.append(tail_fin)
        
        # Add engine exhaust glow
        exhaust_pos_x = tail_pos_x - np.cos(angle) * tail_size/2
        exhaust_pos_y = tail_pos_y - np.sin(angle) * tail_size/2
        
        exhaust = patches.Circle(
            (exhaust_pos_x, exhaust_pos_y),
            size/6,
            facecolor='#FFAA00',
            alpha=0.7,
            zorder=8
        )
        
        drone_parts.append(exhaust)
        
        # Return the collection of parts
        return drone_parts
    
    def render(self, episode, step):
        """Update the visualization with current environment state.
        
        This is the main rendering function that updates all visual elements based on
        the current state of the environment. Key visualization features include:
        - Obstacles displayed 10x larger than physical size for better visibility
        - UAV trajectory paths (black lines) are disabled for cleaner visuals
        - Red polygon shows target encirclement by UAVs
        
        Args:
            episode: Current episode number
            step: Current step number within the episode
        """
        # Update task state information display in top-left corner
        if hasattr(self.env, 'task_state') and self.env.task_state:
            task_state_str = f"Episode: {episode}, Step: {step}, Task: {self.env.task_state.capitalize()}"
            self.task_state_text.set_text(task_state_str)
        
        # Update UAV positions, trajectories, and capture circles
        for i, uav in enumerate(self.env.uavs):
            # Update capture circle position
            self.uav_capture_circles[i].center = (uav.position[0], uav.position[1])
            
            # Intentionally disabled trajectory updates as requested by user
            # By setting empty data arrays, we maintain code structure but don't show any paths
            # This provides a cleaner visual representation without the black trails
            # behind UAVs that were previously cluttering the display
            self.uav_trajectory_plots[i].set_data([], [])
            
            # Update UAV drone patch
            # Remove old drone parts
            for part in self.uav_patches[i]:
                if isinstance(part, patches.Circle) or isinstance(part, patches.Polygon):
                    part.remove()
            
            # Create new drone with updated position and orientation
            drone_parts = self._create_drone(uav.position, uav.yaw, 'black')
            # Add each drone part to the plot
            for part in drone_parts:
                if isinstance(part, patches.Circle) or isinstance(part, patches.Polygon):
                    self.ax.add_patch(part)
            # Store the collection of parts
            self.uav_patches[i] = drone_parts
        
        # Update target position and trajectory
        # For Circle patch, we simply update the center
        self.target_patch.center = (self.env.target.position[0], self.env.target.position[1])
        # Note: No need to update angle for a circle as rotation doesn't affect appearance
        
        # Update obstacle radar scan animations
        for i, obstacle_plot in enumerate(self.obstacle_plots):
            # Get the physical obstacle from environment
            obstacle = self.env.obstacles[i]
            
            # Update circle patch position
            # Note: obstacles are displayed at 10x physical size for better visibility
            # This visual enlargement matches the physics calculations in obstacle avoidance
            obstacle_plot.center = (obstacle.position[0], obstacle.position[1])
            
            # Update radar scan angle - rotate clockwise
            self.obstacle_scan_angles[i] = (self.obstacle_scan_angles[i] + 0.12) % (2 * np.pi)
            
            visual_radius = obstacle_plot.get_radius()
            scan_length = visual_radius
            scan_angle = self.obstacle_scan_angles[i]
            
            # Calculate new endpoint that is guaranteed to be inside the obstacle
            end_x = obstacle_plot.center[0] + scan_length * np.cos(scan_angle)
            end_y = obstacle_plot.center[1] + scan_length * np.sin(scan_angle)
            
            # Update scan line position - connecting center to inside point
            self.obstacle_radar_rings[i].set_data(
                [obstacle_plot.center[0], end_x],
                [obstacle_plot.center[1], end_y]
            )
        
        # Update encirclement visualization with RED polygon as specifically requested
        # This shows how the UAVs are surrounding/encircling the target
        if len(self.env.uavs) >= 3:  # Need at least 3 UAVs to form a polygon
            # Get current UAV positions for polygon vertices
            hull_x = [uav.position[0] for uav in self.env.uavs]
            hull_y = [uav.position[1] for uav in self.env.uavs]
            
            if self.hull_poly is None:
                # Create initial encirclement polygon if not yet created
                self.hull_poly = patches.Polygon(
                    np.column_stack([hull_x, hull_y]),
                    closed=True,
                    fill=True,
                    facecolor='red',  # Red color as explicitly requested by user
                    edgecolor='red',
                    alpha=0.3,        # Semi-transparent
                    zorder=1          # Below UAVs in rendering order
                )
                self.ax.add_patch(self.hull_poly)
            else:
                # Update existing polygon with new UAV positions
                self.hull_poly.set_xy(np.column_stack([hull_x, hull_y]))
        
        # Redraw the canvas and update the display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Introduce a small delay for smoother animation during real-time viewing
        time.sleep(self.frame_interval)
        
        # Capture frame for animation
        canvas = self.fig.canvas
        width, height = self.fig.get_size_inches() * self.fig.get_dpi()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(int(height), int(width), 3)
        self.stored_frames.append(image)
    
    def save_animation(self, filename='simulation.mp4', fps=5):
        """Save the animation to a video file with higher frame rate for smooth playback."""
        if len(self.stored_frames) == 0:
            print("No frames to save!")
            return
            
        # Store the original figure reference
        orig_figure = self.fig
            
        print(f"Saving animation with {len(self.stored_frames)} frames...")
        
        try:
            # Use FFMpegWriter with higher quality settings without creating new windows
            writer = FFMpegWriter(
                fps=fps,  # Higher fps for smoother animation
                metadata=dict(title='Multi-UAV Roundup Simulation', artist='Simulation Tool'),
                bitrate=8000,  # Higher bitrate for better quality
                extra_args=['-pix_fmt', 'yuv420p']  # For compatibility
            )
            
            # Create a new figure for the animation but don't display it on screen
            plt.ioff()  # Turn interactive mode off to prevent window from appearing
            fig_anim = plt.figure(figsize=(12, 12), dpi=150)
            ax_anim = fig_anim.add_subplot(111)
            ax_anim.axis('off')
            
            # Plot the first frame
            img_obj = ax_anim.imshow(self.stored_frames[0])
            
            # Save the animation quietly without showing UI
            with writer.saving(fig_anim, filename, dpi=150):
                for frame in self.stored_frames:
                    img_obj.set_data(frame)
                    writer.grab_frame()
            
            plt.close(fig_anim)
            plt.ion()  # Turn interactive mode back on
            
        except Exception as e:
            print(f"Error saving animation with FFmpeg: {e}")
            
            # Fallback method
            try:
                print("Trying alternative method...")
                
                # Turn interactive mode off to prevent window from appearing
                plt.ioff()
                
                # Try using a more basic writer
                plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg'  # Make sure ffmpeg is in PATH
                
                fig_anim = plt.figure(figsize=(12, 12), dpi=150)
                ax_anim = fig_anim.add_subplot(111)
                ax_anim.axis('off')
                
                # Plot the first frame - IMPORTANT: define img_obj first
                img_obj = ax_anim.imshow(self.stored_frames[0])
                
                # Function to update frames
                def update_frame(frame_num):
                    img_obj.set_data(self.stored_frames[frame_num])
                    return [img_obj]
                
                # Create animation with smooth frame rate
                ani = FuncAnimation(
                    fig_anim, 
                    update_frame, 
                    frames=len(self.stored_frames),
                    interval=1000/fps,  # Convert fps to milliseconds per frame
                    blit=True
                )
                
                # Save animation quietly
                ani.save(filename, writer='ffmpeg', fps=fps, dpi=150,
                        extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
                
                plt.close(fig_anim)
                plt.ion()  # Turn interactive mode back on
                print(f"Animation saved to {filename} (alternative method)")
                
            except Exception as e2:
                print(f"Error with alternative method: {e2}")
                print("Trying most basic method...")
                
                # Most basic method using pillow
                try:
                    fig_anim = plt.figure(figsize=(12, 12), dpi=150)
                    ax_anim = fig_anim.add_subplot(111)
                    ax_anim.axis('off')
                    
                    img_obj = ax_anim.imshow(self.stored_frames[0])
                    
                    ani = FuncAnimation(
                        fig_anim, 
                        lambda i: img_obj.set_data(self.stored_frames[i]), 
                        frames=len(self.stored_frames),
                        interval=1000/fps
                    )
                    
                    ani.save(filename, writer='pillow', fps=fps)
                    plt.close(fig_anim)
                    print(f"Animation saved using basic method to {filename}")
                    
                except Exception as e3:
                    print(f"All animation saving methods failed: {e3}")
                    print("Saving individual frames instead...")
                    
                    # Save individual frames as a last resort
                    frames_dir = os.path.splitext(filename)[0] + "_frames"
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    for i, frame in enumerate(self.stored_frames):
                        from PIL import Image
                        Image.fromarray(frame).save(f"{frames_dir}/frame_{i:04d}.png")
                    
                    print(f"Saved {len(self.stored_frames)} individual frames to {frames_dir}")
                    plt.ion()  # Turn interactive mode back on
                
        # Just provide console output instead of creating a new window
        print(f"\nAnimation successfully saved to:\n{os.path.abspath(filename)}")
    
    def save_image(self, filename='simulation_snapshot.png'):
        """Save the current visualization as an image."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor=self.fig.get_facecolor())
        print(f"Image saved to {filename}")
    
    def close(self):
        """Close the visualization."""
        # Display a completion message on the figure before closing
        try:
            # Display a final message on the plot
            self.ax.set_title('Simulation complete! Close this window to exit.', fontsize=16, color='white')
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            # Only close if show() isn't called
            plt.close(self.fig)
        except Exception as e:
            print(f"Error closing visualization: {e}")