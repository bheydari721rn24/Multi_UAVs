import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import matplotlib.patheffects as pe
from environment import Environment
from utils import CONFIG


class RadarVisualizer:
    """Unified green radar-style visualizer for Multi-UAV simulation"""
    def __init__(self, env: Environment, save_animation=False, fps=10):
        self.env = env
        self.save_animation = save_animation
        self.fps = fps
        # scenario dimensions
        self.width = CONFIG.get("scenario_width", 10.0)
        self.height = CONFIG.get("scenario_height", 10.0)
        # create figure
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.patch.set_facecolor('#000000')
        self.ax.set_facecolor('#003300')
        self.setup_plot()
        # container for animation
        self.ani = None

    def setup_plot(self):
        # limits and labels
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Position (km)', color='white')
        self.ax.set_ylabel('Y Position (km)', color='white')
        # custom grid
        # major
        for x in np.arange(0, self.width+1, 1.0):
            self.ax.axvline(x, color='#00ff00', linestyle='-', alpha=0.3)
            self.ax.axhline(x, color='#00ff00', linestyle='-', alpha=0.3)
        # minor
        for x in np.arange(0, self.width+0.25, 0.25):
            if abs(x % 1.0)>1e-3:
                self.ax.axvline(x, color='#00ff00', linestyle='-', alpha=0.15)
                self.ax.axhline(x, color='#00ff00', linestyle='-', alpha=0.15)
        # micro
        for x in np.arange(0, self.width+0.1, 0.1):
            if abs(x % 0.25)>1e-3:
                self.ax.axvline(x, color='#00ff00', linestyle='-', alpha=0.05)
                self.ax.axhline(x, color='#00ff00', linestyle='-', alpha=0.05)
        # diagonal
        cx, cy = self.width/2, self.height/2
        for off in np.arange(-self.width, self.width, 2):
            self.ax.plot([max(0,off), min(self.width, off+self.height)],
                         [max(0,-off), min(self.height, self.width-off)],
                         color='#00ff00', alpha=0.1)
        for off in np.arange(0, self.width+self.height, 2):
            self.ax.plot([max(0,self.width-off), min(self.width,2*self.width-off)],
                         [max(0,off-self.width), min(self.height, off)],
                         color='#00ff00', alpha=0.1)
        # concentric circles
        self.radar_circles = []
        for r in [2,4,6,8,10]:
            c = Circle((cx, cy), r, fill=False, edgecolor='#00ff00', linestyle='--', alpha=0.3)
            self.ax.add_patch(c)
            self.radar_circles.append(c)
        # boundary
        b = Rectangle((0,0), self.width, self.height, fill=False, edgecolor='#00ff00', linewidth=2.5,
                      path_effects=[pe.withStroke(linewidth=4, foreground='#003300')])
        self.ax.add_patch(b)
        # dynamic elements init
        self.uav_scatter = self.ax.scatter([], [], c='white', s=50, marker='^', zorder=5)
        self.capture_circle = Circle((0,0), CONFIG.get('capture_distance',1.5),
                                     edgecolor='#00ff00', facecolor='none', linestyle='--', alpha=0.7)
        self.ax.add_patch(self.capture_circle)
        self.target_scatter = self.ax.scatter([], [], c='red', s=100, marker='o', edgecolors='white', zorder=6)
        self.status_text = self.ax.text(0.02,0.95,'', transform=self.ax.transAxes,
                                       color='#00ff00', bbox=dict(facecolor='black', edgecolor='#00ff00', pad=4))

    def update_plot(self, frame):
        # step env
        actions = [np.random.uniform(-1,1,2) for _ in self.env.uavs]
        reward = self.env.step(actions)
        # update UAVs
        pos = np.array([u.position for u in self.env.uavs])
        self.uav_scatter.set_offsets(pos)
        # update target
        tp = self.env.target.position
        self.target_scatter.set_offsets([tp])
        # capture
        self.capture_circle.center = tp
        # status
        self.status_text.set_text(f"Step: {frame}")
        # return artists
        arts = [self.uav_scatter, self.target_scatter, self.capture_circle, self.status_text] + self.radar_circles
        return arts

    def run_simulation(self, steps=100):
        self.ani = animation.FuncAnimation(self.fig, self.update_plot,
                                           frames=steps, interval=100, blit=True)
        if self.save_animation:
            out = f"./plots/radar_{np.random.randint(1e6)}.mp4"
            self.ani.save(out, fps=self.fps)
            print(f"Saved animation to {out}")
        else:
            plt.show()


if __name__ == '__main__':
    env = Environment(num_uavs=5, num_obstacles=6, enable_ahfsi=False)
    env.reset()
    viz = RadarVisualizer(env)
    viz.run_simulation(steps=50)
