from control.mppi_isaac.mppiisaac.utils.conversions import quaternion_to_yaw

import rospy
import numpy as np
import networkx as nx

from itertools import combinations

from shapely import buffer
from shapely.affinity import rotate
from shapely.geometry import Point, LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union


class NVG:

    def __init__(self, range_x, range_y, mass_threshold, path_inflation):
        self.range_x = range_x
        self.range_y = range_y

        self.mass_threshold = mass_threshold
        self.path_inflation = path_inflation
    
    def graph(self, q_init, q_goal, actors):
        actor_polygons, _ = self.generate_polygons(actors)
        avoid_obstacle, _ = self.generate_polygons(actors, self.path_inflation - 1e-2)

        graph = nx.Graph()
        self.add_node_to_graph(graph, (*q_init, 0.), avoid_obstacle.values())

        actor_nodes = self.generate_nodes(list(actor_polygons.values()))
        for node in actor_nodes:
            self.add_node_to_graph(graph, node, avoid_obstacle.values())

        self.add_node_to_graph(graph, (*q_goal, 0.), avoid_obstacle.values())

        try:
            nx.shortest_path(graph, source=0, target=len(graph.nodes) -1)
        except nx.NetworkXNoPath:
            rospy.loginfo("Avoidance is not possible.")
            pass

        return graph

    def add_node_to_graph(self, graph, new_node, polygons):
        new_node_index = len(graph.nodes)
        graph.add_node(new_node_index, pos=new_node[:2], cost=new_node[2])

        node_connections = 0

        organized_nodes = self.find_closest_nodes(graph, new_node[:2])
        for (node, node_pos) in organized_nodes:
            if node != new_node_index:
                edge_line = LineString([node_pos, new_node[:2]])

                if not any(edge_line.intersects(polygon) for polygon in polygons):
                    graph.add_edge(node, new_node_index, length=edge_line.length)
                    node_connections += 1
        return graph

    def generate_polygons(self, actors, overwrite_inflation=None):
        margin = self.path_inflation if overwrite_inflation is None else overwrite_inflation
        
        masses, shapes = {}, {}

        actor_wrappers, actors_state = actors
        for actor in range(1, len(actor_wrappers)):
            actor_wrapper = actor_wrappers[actor]

            mass = actor_wrapper.mass
            name = actor_wrapper.name
            size = actor_wrapper.size

            obs_pos = actors_state[actor, :2]
            obs_rot = quaternion_to_yaw(actors_state[actor, 3:7])

            corners = [
                (obs_pos[0] - size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] - size[1] / 2),
                (obs_pos[0] + size[0] / 2, obs_pos[1] + size[1] / 2),
                (obs_pos[0] - size[0] / 2, obs_pos[1] + size[1] / 2)
            ]

            polygon = Polygon(corners)
            polygon = rotate(polygon, obs_rot, use_radians=True)
            polygon = buffer(polygon, margin, cap_style='flat', join_style='bevel')

            shapes[name] = polygon
            masses[name] = mass

        return shapes, masses

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
    def get_corner_points(polygons, range_x, range_y):
        free_polygons = NVG.get_free_areas(polygons, range_x, range_y)

        corner_nodes = []
        for polygon in free_polygons:
            for corner in polygon.exterior.coords[:-1]:
                corner_nodes.append(corner)

        return np.array(corner_nodes)

    @staticmethod
    def get_free_areas(polygons, range_x, range_y):
        occupied_area = unary_union(polygons)
        
        xmin, xmax = range_x
        ymin, ymax = range_y
        total_area = box(xmin, ymin, xmax, ymax)
        
        free_area = total_area.difference(occupied_area)
        
        if isinstance(free_area, Polygon):
            free_areas = [free_area]
        elif isinstance(free_area, MultiPolygon):
            free_areas = list(free_area.geoms)
        else:
            free_areas = []
        
        return free_areas


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
    def add_passages_to_graph(graph, passages, shapes=None):
        num_existing_nodes = graph.number_of_nodes()

        for i, (x, y, cost, search_distance) in enumerate(passages):
            new_node_index = num_existing_nodes + i
            graph.add_node(new_node_index, pos=(x, y), cost=cost)

            new_node_pos = (x, y)
            for node, node_pos in graph.nodes(data='pos'):
                edge_line = LineString([node_pos, new_node_pos])
                if node != new_node_index and edge_line.length <= search_distance:
                    if not shapes: 
                        graph.add_edge(new_node_index, node, length=edge_line.length)
                    if shapes and not any(edge_line.intersects(polygon) for polygon in shapes.values()):
                        graph.add_edge(new_node_index, node, length=edge_line.length)
        return graph

    @staticmethod    
    def search_distance_radius(polygon):
        vertices = list(polygon.exterior.coords)
        distance = [LineString([v_1, v_2]).length for v_1, v_2 in combinations(vertices, 2)]
        return max(distance)
