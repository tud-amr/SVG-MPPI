import rospy
import numpy as np
import networkx as nx

from collections import defaultdict
from itertools import combinations

from tf.transformations import euler_from_quaternion
from shapely import buffer, prepare
from shapely.affinity import rotate
from shapely.geometry import Point, MultiPoint, LineString, MultiPolygon, Polygon, box
from shapely.ops import nearest_points, unary_union


class SVG:

    def __init__(self, range_x, range_y, mass_threshold, path_inflation):
        self.range_x = range_x
        self.range_y = range_y

        self.mass_threshold = mass_threshold
        self.path_inflation = path_inflation

    def graph(self, q_init, q_goal, actors):
        actor_polygons, masses = self.generate_polygons(actors)
        avoid_obstacle, _ = self.generate_polygons(actors, self.path_inflation - 1e-3)

        graph = nx.Graph()
        graph = self.add_node_to_graph(graph, (*q_init, 0.), avoid_obstacle.values())

        actor_nodes = self.generate_nodes(actor_polygons)
        for node in actor_nodes:
            graph = self.add_node_to_graph(graph, node, avoid_obstacle.values())

        graph = self.add_node_to_graph(graph, (*q_goal, 0.), avoid_obstacle.values())
        try:
            nx.shortest_path(graph, source=0, target=len(graph.nodes) - 1)
        except nx.NetworkXNoPath:
            rospy.loginfo("Avoidance is not possible, creating passage nodes.")
            pass

        non_inflated_shapes, _ = self.generate_polygons(actors, 0.)
        passage_nodes = self.generate_passages(non_inflated_shapes, masses)

        graph = self.add_passage_to_graph(graph, passage_nodes, non_inflated_shapes.values())
        return graph

    def add_node_to_graph(self, graph, new_node, polygons=None, knn=None, connect_to_last=False):
        new_node_index = len(graph.nodes)
        last_node_index = len(graph.nodes) - 1

        graph.add_node(new_node_index, pos=new_node[:2], cost=new_node[2])

        if connect_to_last:
            last_node_pos = graph.nodes(data='pos')[last_node_index]
            edge_line = LineString([last_node_pos, new_node[:2]])

            if polygons is not None:
                if not any(edge_line.intersects(polygon) for polygon in polygons):
                    graph.add_edge(new_node_index, last_node_index, length=edge_line.length)
            else:
                graph.add_edge(new_node_index, last_node_index, length=edge_line.length)

        node_connections = 0
        organized_nodes = self.find_closest_nodes(graph, new_node[:2])
        for (node, node_pos) in organized_nodes:
            if connect_to_last and node == last_node_index:
                continue 

            if node != new_node_index:
                edge_line = LineString([node_pos, new_node[:2]])

                if polygons is not None:
                    if not any(edge_line.intersects(polygon) for polygon in polygons):
                        graph.add_edge(node, new_node_index, length=edge_line.length)
                        node_connections += 1
                else:
                    graph.add_edge(node, new_node_index, length=edge_line.length)
                    node_connections += 1

            if knn and node_connections >= knn:
                break
            
        return graph

    def add_passage_to_graph(self, graph, passages, polygons=None):
        passage_dict = defaultdict(list)
        for (x, y, cost, passage_identifier) in passages:
            passage_dict[passage_identifier].append((x, y, cost))
        
        for passage_identifier, nodes in passage_dict.items():
            if len(nodes) > 0:
                entry_node = nodes[0]
                graph = self.add_node_to_graph(graph, entry_node, polygons, knn=4)

            if len(nodes) > 1:
                exit_node = nodes[1]
                graph = self.add_node_to_graph(graph, exit_node, polygons, knn=4, connect_to_last=True)

            if len(nodes) > 2:
                entry_node_index = list(graph.nodes)[-2]
                exit_node_index = list(graph.nodes)[-1]

                if not graph.has_edge(entry_node_index, exit_node_index):
                    self.add_node_to_graph(graph, nodes[2], polygons, knn=4, connect_to_last=True)
        return graph

    def generate_polygons(self, actors, overwrite_inflation=None):
        margin = overwrite_inflation if overwrite_inflation is not None else self.path_inflation

        masses, shapes = {}, {}

        actor_wrappers, actors_state = actors
        for actor in range(1, len(actor_wrappers)):
            actor_wrapper = actor_wrappers[actor]

            mass = actor_wrapper.mass
            name = actor_wrapper.name
            size = actor_wrapper.size

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
            polygon = buffer(polygon, margin, cap_style='flat', join_style='bevel')

            shapes[name] = polygon
            masses[name] = mass

        return shapes, masses

    def generate_nodes(self, shapes):
        nodes = np.empty((0, 3), dtype='float')

        if shapes.values():
            corner_points = self.get_corner_points(list(shapes.values()), self.range_x, self.range_y)
            if len(corner_points) != 0:
                corner_points = np.hstack((corner_points, np.zeros((corner_points.shape[0], 1))))
                nodes = np.vstack((nodes, corner_points))

            nodes = self.filter_nodes(nodes, shapes.values(), self.range_x, self.range_y)
        return nodes

    def generate_passages(self, shapes, masses):
        passages = np.empty((0, 4), dtype='float')

        obstacles_id_pairs = list(combinations(shapes.keys(), 2))
        passage_identifier = 0

        for id_1, id_2 in obstacles_id_pairs:
            heavy_id = id_1 if masses[id_1] >= masses[id_2] else id_2
            heavy_ob = shapes[heavy_id]

            light_id = id_1 if masses[id_1] < masses[id_2] else id_2
            light_ob = shapes[light_id]

            heavy_mass = masses[heavy_id]
            light_mass = masses[light_id]

            light_ob = light_ob.convex_hull
            heavy_ob = heavy_ob.convex_hull

            prepare(light_ob)
            prepare(heavy_ob)

            light_point, heavy_point = nearest_points(light_ob, heavy_ob)
            shortest_line = LineString([light_point, heavy_point])
            minimum_distance = shortest_line.length

            # Exception 1: Both masses exceed the maximum
            if masses[id_1] >= self.mass_threshold and masses[id_2] >= self.mass_threshold:
                continue

            # Exception 2: The distance between the obstacles is already sufficient
            if minimum_distance > (2 * self.path_inflation):
                continue

            # Exception 3: obstacles overlap as convex shapes
            if light_ob.intersects(heavy_ob):
                continue

            points_list = self.get_nearest_points_excluding_vertices(heavy_ob, light_ob)
            convex_hull = MultiPoint(points_list).convex_hull

            coords = list(convex_hull.exterior.coords) if convex_hull.geom_type == 'Polygon' else list(convex_hull.coords)
            for coord_idx in range(len(coords) - 1):
                p1, p2 = Point(coords[coord_idx]), Point(coords[coord_idx + 1])

                if self.is_point_in_shape(p1, light_ob) and self.is_point_in_shape(p2, light_ob):
                    continue

                if self.is_point_in_shape(p1, heavy_ob) and self.is_point_in_shape(p2, heavy_ob):
                    continue

                boundary = LineString([coords[coord_idx], coords[coord_idx + 1]])

                mass_p1 = heavy_mass if self.is_point_in_shape(Point(boundary.coords[0]), heavy_ob) else light_mass
                
                interpolation_distance = boundary.length / (light_mass + heavy_mass) * mass_p1

                if interpolation_distance > boundary.length:
                    interpolation_distance = boundary.length

                if interpolation_distance > self.path_inflation:
                    interpolation_distance = self.path_inflation

                passage_point = boundary.interpolate(interpolation_distance)
                
                light_ob_distance = passage_point.distance(light_ob)
                heavy_ob_distance = passage_point.distance(heavy_ob)
            
                light_ob_cost = max(0, 1 - (light_ob_distance / self.path_inflation)) * light_mass
                heavy_ob_cost = max(0, 1 - (heavy_ob_distance / self.path_inflation)) * heavy_mass

                heavy_ob_cost = heavy_ob_cost if heavy_mass < self.mass_threshold else 0.
            
                passage_cost = light_ob_cost + heavy_ob_cost
                passage_point = boundary.interpolate(interpolation_distance)

                passages = np.vstack((passages, (passage_point.x, passage_point.y, passage_cost, passage_identifier)))

            if convex_hull.geom_type == 'Polygon':
                heavy_mass, light_mass = masses[heavy_id], masses[light_id]
                interpolation_distance = shortest_line.length / (light_mass + heavy_mass) * light_mass

                if masses[id_1] >= self.mass_threshold or masses[id_2] >= self.mass_threshold:
                    if shortest_line.length < self.path_inflation and mass_p1 == light_mass:
                        interpolation_distance = boundary.length - self.path_inflation

                if interpolation_distance > self.path_inflation:
                    interpolation_distance = self.path_inflation

                if interpolation_distance > shortest_line.length:
                    interpolation_distance = shortest_line.length

                bridge_point = shortest_line.interpolate(interpolation_distance)
                passages = np.vstack((passages, (bridge_point.x, bridge_point.y, 0., passage_identifier)))

            passage_identifier += 1

        return passages

    @staticmethod
    def get_corner_points(polygons, range_x, range_y):
        free_polygons = SVG.get_free_areas(polygons, range_x, range_y)

        corner_nodes = []
        for polygon in free_polygons:
            for corner in polygon.exterior.coords[:-1]:
                corner_nodes.append(corner)
            for interior in polygon.interiors:
                for corner in interior.coords[:-1]:
                    corner_nodes.append(corner)

        return np.array(corner_nodes)

    @staticmethod
    def get_free_areas(polygons, range_x, range_y):
        convex_polygons = [polygon.convex_hull for polygon in polygons]
        occupied_area = unary_union(convex_polygons)
        
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
    def get_intersection_points(shapes, masses, mass_threshold):
        intersection_points = []

        obstacles_id_pairs = list(combinations(shapes.keys(), 2))
        for id_1, id_2 in obstacles_id_pairs:
            if masses[id_1] >= mass_threshold and masses[id_2] >= mass_threshold:
                continue
            
            polygon_i, polygon_j = shapes[id_1], shapes[id_2]
            intersection = polygon_i.boundary.intersection(polygon_j.boundary)

            if isinstance(intersection, Point):
                intersection_points.append([intersection.x, intersection.y])

            elif isinstance(intersection, LineString):
                for point in intersection.coords:
                    intersection_points.append([point[0], point[1]])

            elif isinstance(intersection, Polygon):
                for point in intersection.exterior.coords:
                    intersection_points.append([point[0], point[1]])
            else:
                for point in intersection.geoms:
                    intersection_points.append([point.x, point.y])

        return np.array(intersection_points)

    @staticmethod
    def get_nearest_points_excluding_vertices(polygon_1, polygon_2):
        def is_vertex(point, polygon):
            return any(point.equals(Point(vertex)) for vertex in polygon.exterior.coords)
        
        nearest_points_set = set()
        
        for point in polygon_1.exterior.coords:
            if not is_vertex(Point(point), polygon_2):
                nearest_point = nearest_points(Point(point), polygon_2)[1]
                nearest_points_set.add(nearest_point)
        
        for point in polygon_2.exterior.coords:
            if not is_vertex(Point(point), polygon_1):
                nearest_point = nearest_points(Point(point), polygon_1)[1]
                nearest_points_set.add(nearest_point)

        return list(nearest_points_set)

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
    def filter_nodes(nodes, polygons, range_x, range_y):
        filtered_nodes = []

        for node in nodes:
            point = Point(node)
            if not range_x[0] <= point.x <= range_x[1]:
                continue

            if not range_y[0] <= point.y <= range_y[1]:
                continue

            is_within_polygon = any(polygon.contains(point) for polygon in polygons)
            if not is_within_polygon:
                filtered_nodes.append(node)

        return filtered_nodes

    @staticmethod
    def is_point_in_shape(point, shape, epsilon=1e-3):
        if point.within(shape):
            return True

        if shape.boundary.distance(point) < epsilon:
            return True

        return False

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return euler_from_quaternion(quaternion)[-1]
