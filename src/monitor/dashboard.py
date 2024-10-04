import roslib
import yaml
import shapely

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from tf.transformations import euler_from_quaternion
from shapely.affinity import rotate


plt.ion()


class Dashboard:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, range_x, range_y):
        self.fig_overview, self.ax_1 = None, None
        self.fig_planning, self.ax_2 = None, None
        self.fig_rollouts, self.ax_3 = None, None
        self.fig_progress, self.ax_4 = None, None

        self.range_x = range_x
        self.range_y = range_y

    @classmethod
    def create_dashboard(cls, layout):
        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config = yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config = yaml.safe_load(stream)

        params = {**base_config, **world_config}

        range_x = params['range_x']
        range_y = params['range_y']
        
        return cls(range_x, range_y)

    def update_overview(self, actors, states):
        if self.fig_overview is None:
            self.fig_overview, self.ax_1 = plt.subplots()

        self.ax_1.clear()

        for actor, state in zip(actors, states):
            position, orientation = state[:3], state[3:7]

            size_x, size_y, _ = actor.size
            obs_pos = position[:2]

            corners = np.array([[-size_x / 2, -size_y / 2],
                                [size_x / 2, -size_y / 2],
                                [size_x / 2, size_y / 2],
                                [-size_x / 2, size_y / 2],
                                [-size_x / 2, -size_y / 2]])

            obs_rot = self.quaternion_to_yaw(orientation)
            rotation_matrix = np.array([[np.cos(obs_rot), -np.sin(obs_rot)],
                                        [np.sin(obs_rot), np.cos(obs_rot)]])

            rotate_corners = np.dot(corners, rotation_matrix)
            translate_corners = np.add(rotate_corners, obs_pos)

            polygon = Polygon(translate_corners,
                              closed=True, color=actor.color)
            self.ax_1.add_patch(polygon)

        self.ax_1.set_xlim(*self.range_x)
        self.ax_1.set_ylim(*self.range_y)

        self.ax_1.set_title('Top View Overview')
        self.ax_1.set_xlabel('X Position')
        self.ax_1.set_ylabel('Y Position')
        self.ax_1.set_aspect('equal')

        self.fig_overview.canvas.draw()
        self.fig_overview.canvas.flush_events()

    def update_planning(self, actors, shortest_path=None, graph=None):
        if self.fig_planning is None:
            self.fig_planning, self.ax_2 = plt.subplots()

        self.ax_2.clear()

        for actor, state in zip(*actors):
            size = actor.size
            obs_pos = state[:2]
            obs_rot = self.quaternion_to_yaw(state[3:7])

            corners = [
                (obs_pos[0] - size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] + size[1] / 2),
                (obs_pos[0] - size[0] / 2, obs_pos[1] + size[1] / 2)
            ]

            polygon = shapely.Polygon(corners)
            polygon = rotate(polygon, obs_rot, use_radians=True)

            patch_polygon = Polygon(polygon.exterior.coords, closed=True, color=actor.color)
            self.ax_2.add_patch(patch_polygon)

        if graph is not None:
            nodes = np.array([(*data['pos'], data['cost']) for _, data in graph.nodes(data=True)])
            edges = np.array([(graph.nodes[u]['pos'], graph.nodes[v]['pos']) for u, v, _ in graph.edges(data=True)])

            node_scatter = self.ax_2.scatter(nodes[:, 0], nodes[:, 1], c=nodes[:, 2], cmap=plt.cm.viridis)
            cbar = plt.colorbar(node_scatter, ax=self.ax_2)
            cbar.set_label('Cost')

            for edge in edges:
                self.ax_2.plot(edge[:, 0], edge[:, 1], color='blue', linewidth=0.1)
        
        if shortest_path is not None:
            self.ax_2.plot(shortest_path[:, 0], shortest_path[:, 1], 'g-o', linewidth=2)

        self.ax_2.set_title('Global path planning')
        self.ax_2.autoscale(enable=True)
        self.ax_2.set_aspect('equal')

        self.ax_2.set_xlabel('X Position')
        self.ax_2.set_ylabel('Y Position')

        self.fig_planning.canvas.draw()
        self.fig_planning.canvas.flush_events()
        plt.pause(3)

    def update_rollouts(self, rollout_states, best_states):
        if self.fig_rollouts is None:
            self.fig_rollouts, self.ax_3 = plt.subplots()

        self.ax_3.clear()

        rollout_states, best_states = rollout_states.cpu(), best_states.cpu()
        if rollout_states.numel() > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, rollout_states.shape[0]))

            for i in range(rollout_states.shape[0]):
                x_values, y_values = rollout_states[i, :, 0], rollout_states[i, :, 2]
                self.ax_3.plot(x_values, y_values, color=colors[i], alpha=0.4)

            x_values_best, y_values_best = best_states[0, :, 0], best_states[0, :, 2]
            self.ax_3.plot(x_values_best, y_values_best, color='red', alpha=1.0)

        self.ax_3.set_title('Robot DOF rollouts')
        self.ax_3.set_aspect('equal')

        self.ax_3.autoscale(enable=True)

        self.fig_rollouts.canvas.draw()
        self.fig_rollouts.canvas.flush_events()
        plt.pause(0.001)

    def update_progress(self, dofs, goal):
        if self.fig_progress is None:
            self.fig_progress, self.ax_4 = plt.subplots()

        self.ax_4.clear()

        self.ax_4.scatter(goal[0], goal[1], color='red', label='Goal')
        self.ax_4.scatter(dofs[0], dofs[2], color='green', label='Robot')

        self.ax_4.set_title('Coordinates')
        self.ax_4.set_xlim(*self.range_x)
        self.ax_4.set_ylim(*self.range_y)

        self.fig_progress.canvas.draw()
        self.fig_progress.canvas.flush_events()

    def destroy(self):
        if self.fig_overview:
            plt.close(self.fig_overview)

        if self.fig_planning:
            plt.close(self.fig_planning)

        if self.fig_rollouts:
            plt.close(self.fig_rollouts)

        if self.fig_progress:
            plt.close(self.fig_progress)

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return euler_from_quaternion(quaternion)[-1]
