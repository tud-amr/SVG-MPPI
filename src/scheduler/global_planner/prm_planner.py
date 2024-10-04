import time
import numpy as np
import networkx as nx

from tf.transformations import euler_from_quaternion
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import rotate
from shapely import buffer


class PRM:

    MAX_ITERATION_TIME = 300
    OBS_SAFETY_MARGINS = 1e-2

    NEAREST_NEIGHBOURS = 4
    CONNECTING_NODES_R = 0.5

    def __init__(self, range_x, range_y, mass_threshold, path_inflation):
        self.range_x = range_x
        self.range_y = range_y

        self.mass_threshold = mass_threshold 
        self.path_inflation = path_inflation
    
    def graph(self, q_init, q_goal, actors, margin=OBS_SAFETY_MARGINS):
        actor_polygons = self.generate_polygons(actors)
        avoid_obstacle = self.generate_polygons(actors, margin)

        nodes = self.generate_nodes(actor_polygons)
        nodes = np.vstack((nodes, [*q_init, 0.]))
        nodes = np.vstack((nodes, [*q_goal, 0.]))

        graph = nx.Graph()
        for node in nodes:
            graph = self.add_node_to_graph(graph, node, avoid_obstacle)

        init_node = self.find_closest_node(graph, q_init)
        goal_node = self.find_closest_node(graph, q_goal)

        node_increment = nodes.shape[0]
        
        start_time = time.time()
        while not nx.has_path(graph, init_node, goal_node):
            new_nodes = np.empty((0, 3), dtype='float')

            while new_nodes.shape[0] < node_increment:
                new_node = self.generate_random_node()

                if not self.is_node_in_obstacle_space(new_node, avoid_obstacle):
                    new_nodes = np.vstack((new_nodes, new_node))

                if (time.time() - start_time) > self.MAX_ITERATION_TIME:
                    return graph

            for new_node in new_nodes:
                graph = self.add_node_to_graph(graph, new_node, avoid_obstacle)

            if (time.time() - start_time) > self.MAX_ITERATION_TIME:
                break

            init_node = self.find_closest_node(graph, q_init)
            goal_node = self.find_closest_node(graph, q_goal)

        return graph

    def add_node_to_graph(self, graph, new_node, polygons, knn=NEAREST_NEIGHBOURS, r=CONNECTING_NODES_R):
        new_node_index = len(graph.nodes)
        graph.add_node(new_node_index, pos=new_node[:2], cost=new_node[2])

        node_connections = 0

        organized_nodes = self.find_closest_nodes(graph, new_node[:2])
        for (node, node_pos) in organized_nodes:
            if node != new_node_index:
                edge_line = LineString([node_pos, new_node[:2]])

                if edge_line.length > r:
                    break

                if not any(edge_line.intersects(polygon) for polygon in polygons):
                    graph.add_edge(node, new_node_index, length=edge_line.length)
                    node_connections += 1

            if node_connections >= knn:
                break

        return graph

    def generate_random_node(self):
        rand_x = np.random.uniform(*self.range_x)
        rand_y = np.random.uniform(*self.range_y)
        return np.array([rand_x, rand_y, 0.])

    def generate_polygons(self, actors, overwrite_inflation=None):
        margin = overwrite_inflation if overwrite_inflation is not None else self.path_inflation

        shapes = []
        actor_wrappers, actors_state = actors
        for actor in range(1, len(actor_wrappers)):
            actor_wrapper = actor_wrappers[actor]

            mass = actor_wrapper.mass
            size = actor_wrapper.size

            inflation = margin
            if mass > self.mass_threshold:
                inflation = self.path_inflation

            obs_pos = actors_state[actor, :2]
            obs_rot = self.quaternion_to_yaw(actors_state[actor, 3:7])

            corners = [
                (obs_pos[0] - size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] + size[1] / 2),
                (obs_pos[0] - size[0] / 2, obs_pos[1] + size[1] / 2)
            ]

            polygon = Polygon(corners)
            polygon = rotate(polygon, obs_rot, use_radians=True)
            polygon = buffer(polygon, inflation, cap_style='flat', join_style='bevel')

            shapes.append(polygon) 

        return shapes

    def generate_nodes(self, polygons):
        nodes = np.empty((0, 3), dtype='float')

        if polygons:
            corner_points = self.get_corner_points(polygons, self.range_x, self.range_y)
            if len(corner_points) != 0:
                corner_points = np.hstack((corner_points, np.zeros((corner_points.shape[0], 1))))
                nodes = np.vstack((nodes, corner_points))

            nodes = self.filter_nodes(nodes, polygons)

        return nodes

    @staticmethod
    def get_corner_points(polygons, x_limit, y_limit):
        corner_nodes = []
        for polygon in polygons:
            for corner in polygon.exterior.coords[:-1]:
                if x_limit[0] <= corner[0] <= x_limit[1]:
                    if y_limit[0] <= corner[1] <= y_limit[1]:
                        corner_nodes.append(corner)

        return np.array(corner_nodes)

    @staticmethod
    def is_node_in_obstacle_space(node, c_space_obstacles):
        return any(obstacle.contains(Point(*node)) for obstacle in c_space_obstacles)

    @staticmethod
    def filter_nodes(nodes, polygons, radius=0.1):
        filtered_nodes = []

        for node in nodes:
            point = Point(node)

            is_within_polygon = any(polygon.contains(point) for polygon in polygons)
            if not is_within_polygon:
                filtered_nodes.append(node)
            else:
                is_valid = any(not any(polygon.contains(Point(node[0] + dx, node[1] + dy)) for polygon in polygons)
                               for dx in np.arange(-radius, radius + 0.01, 0.1)
                               for dy in np.arange(-radius, radius + 0.01, 0.1))
                if is_valid:
                    filtered_nodes.append(node)

        filtered_nodes = np.array(filtered_nodes)
        return np.unique(filtered_nodes, axis=0)

    @staticmethod
    def find_closest_nodes(graph, coordinate):
        nodes_with_distances = []
        for node, pos in graph.nodes(data='pos'):
            distance = np.linalg.norm(np.array(pos) - np.array(coordinate))
            nodes_with_distances.append((node, pos, distance))

        nodes_with_distances.sort(key=lambda x: x[2])

        sorted_nodes = [(node, pos) for node, pos, _ in nodes_with_distances]
        return sorted_nodes
    
    @staticmethod
    def find_closest_node(graph, coordinate):
        closest_node = None
        min_distance = float('inf')

        for node, pos in graph.nodes(data='pos'):
            distance = np.linalg.norm(np.array(pos) - np.array(coordinate))

            if distance < min_distance:
                closest_node = node
                min_distance = distance

        return closest_node

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return euler_from_quaternion(quaternion)[-1]
