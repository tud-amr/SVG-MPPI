# type: ignore
from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper    # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig            # type: ignore

import io
import copy
import math
import random
import roslib
import rospy
import torch
import time
import yaml
import zerorpc

import numpy as np

import shapely
from shapely.geometry import Polygon
from shapely.affinity import rotate

from scipy.spatial.transform import Rotation
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from isaacgym import gymapi


class SimulateWorld:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, simulation, controller, device):
        self.simulation = simulation
        self.controller = controller

        self.params = params
        self.config = config
        self.device = device

        self.pos_tolerance = params['controller']['pos_tolerance']
        self.yaw_tolerance = params['controller']['yaw_tolerance']
        self.vel_tolerance = params['controller']['vel_tolerance']
        self.replan_timing = params['controller']['replan_timing']

        self._goal = None
        self._mode = None

        self.is_goal_reached = False
        self.replan_watchdog = None

    @classmethod
    def build(cls, config: ExampleConfig, layout: str, use_viewer: bool):
        world = cls.create(config, layout, use_viewer)
        world.configure()
        return world

    @classmethod
    def create(cls, config, layout, use_viewer):
        simulation = IsaacGymWrapper(
            config["isaacgym"],
            init_positions=config["initial_actor_positions"],
            actors=config["actors"],
            num_envs=1,
            viewer=use_viewer,
            device=config["mppi"].device,
        )

        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config = yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config = yaml.safe_load(stream)

        params = {**base_config, **world_config}

        controller = zerorpc.Client(timeout=None, heartbeat=None)
        controller.connect("tcp://127.0.0.1:4242")

        return cls(params, config, simulation, controller, config["mppi"].device)

    def configure(self, additions=None, controller_additions=None, apply_mass_noise=True):
        if additions is None:
            if self.params["random"]:
                additions = self.random_additions()
            else:
                additions = self.create_additions()

        controller_additions = controller_additions if controller_additions else additions

        self.simulation.add_to_envs(additions, apply_mass_noise)
        self.controller.add_to_env(controller_additions, apply_mass_noise)

        init_state = self.params["environment"]["robot"]["init_state"]
        x, y, yaw = init_state[0], init_state[1], init_state[2] * (math.pi / 180.0)

        self.simulation.set_actor_dof_state(torch.tensor([x, 0., y, 0., yaw, 0.], device=self.device))
        self.update_objective(np.array([[x, y]]))

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]
        self.set_viewer(self.simulation._gym, self.simulation.viewer, cam_pos, cam_tar)
        
        self.replan_watchdog = time.time()

    def create_additions(self):
        additions = []

        if self.params["environment"].get("demarcation", None):
            for wall in self.params["environment"]["demarcation"]:
                obs_type = next(iter(wall))
                obs_args = self.params["objects"][obs_type]

                obstacle = {**obs_args, **wall[obs_type]}

                rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
                obstacle["init_ori"] = list(rot)

                additions.append(obstacle)

        if self.params["environment"].get("obstacles", None):
            for obstacle in self.params["environment"]["obstacles"]:
                obs_type = next(iter(obstacle))
                obs_args = self.params["objects"][obs_type]

                obstacle = {**obs_args, **obstacle[obs_type]}

                rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
                obstacle["init_ori"] = list(rot)

                additions.append(obstacle)
        return additions

    def random_additions(self, build_walls=True, additions=None, range_x=None, range_y=None):
        additions = additions if additions is not None else []

        if build_walls:
            additions += self.create_walls()

        range_x = self.params['range_x'] if range_x is None else range_x
        range_y = self.params['range_y'] if range_y is None else range_y

        area = (range_x[1] - range_x[0]) * (range_y[1] - range_y[0])

        stationary_percentage = self.params['stationary_percentage']
        stationary_size_noise = self.params['stationary_size_noise']

        adjustable_percentage = self.params['adjustable_percentage']
        adjustable_size_noise = self.params['adjustable_size_noise']

        inflation = self.params["scheduler"]["path_inflation"]
        init_pose = self.params["environment"]["robot"]["init_state"]
        goal_pose = self.params["goal"]

        excluded_poses = ({'init_pos': init_pose, 'init_ori': [0., 0., 0.], 'size': [2*inflation, 2*inflation]},
                          {'init_pos': goal_pose, 'init_ori': [0., 0., 0.], 'size': [2*inflation, 2*inflation]})

        stationary_area_target = area * stationary_percentage
        adjustable_area_target = area * adjustable_percentage

        current_stationary_area = 0.
        current_adjustable_area = 0.

        while current_stationary_area < stationary_area_target:
            obstacle = copy.deepcopy(self.params["objects"]["stationary"])
            obstacle["name"] = f"Obstacle {len(additions)}"

            random_yaw = random.uniform(-np.pi, np.pi)
            random_x = random.uniform(*range_x)
            random_y = random.uniform(*range_y)

            init_pos = [random_x, random_y, 0.5]
            init_ori = self.yaw_to_quaternion(random_yaw)

            obstacle["init_ori"] = init_ori
            obstacle["init_pos"] = init_pos

            size_x, size_y, size_z = obstacle["size"]
            obstacle["size"][0] = size_x + \
                np.random.uniform(-stationary_size_noise *
                                  size_x, stationary_size_noise * size_x)
            obstacle["size"][1] = size_y + \
                np.random.uniform(-stationary_size_noise *
                                  size_y, stationary_size_noise * size_y)
            obstacle["size"][2] = size_z + \
                np.random.uniform(-stationary_size_noise *
                                  size_z, stationary_size_noise * size_z)

            if not self.is_obstacle_overlapping(init_pos, obstacle["size"], init_ori, additions, excluded_poses):
                current_stationary_area += (obstacle["size"]
                                            [0] * obstacle["size"][1])
                additions.append(obstacle)

        while current_adjustable_area < adjustable_area_target:
            obstacle = copy.deepcopy(self.params["objects"]["adjustable"])
            obstacle["name"] = f"Obstacle {len(additions)}"

            random_yaw = random.uniform(-np.pi, np.pi)
            random_x = random.uniform(*range_x)
            random_y = random.uniform(*range_y)

            init_pos = [random_x, random_y, 0.5]
            init_ori = self.yaw_to_quaternion(random_yaw)

            obstacle["init_ori"] = init_ori
            obstacle["init_pos"] = init_pos

            size_x, size_y, _ = obstacle["size"]
            obstacle["size"][0] = size_x + \
                np.random.uniform(-adjustable_size_noise *
                                  size_x, adjustable_size_noise * size_x)
            obstacle["size"][1] = size_y + \
                np.random.uniform(-adjustable_size_noise *
                                  size_y, adjustable_size_noise * size_y)

            if not self.is_obstacle_overlapping(init_pos, obstacle["size"], init_ori,
                                                additions, excluded_poses):
                current_adjustable_area += (obstacle["size"]
                                            [0] * obstacle["size"][1])
                additions.append(obstacle)

        return additions

    def grid_additions(self):
        additions = self.create_walls()

        range_x = self.params['range_x']
        range_y = self.params['range_y']

        obstacle_size = self.params["objects"]["adjustable"]['size']

        inflation = self.params["scheduler"]["path_inflation"]
        init_pose = self.params["environment"]["robot"]["init_state"]
        goal_pose = self.params["goal"]

        excluded_poses = ({'init_pos': init_pose, 'init_ori': [0., 0., 0.], 'size': [2*inflation, 2*inflation]},
                          {'init_pos': goal_pose, 'init_ori': [0., 0., 0.], 'size': [2*inflation, 2*inflation]})

        x_step = obstacle_size[0] + .49
        y_step = 2 * inflation + 4 * obstacle_size[1]

        start_x, end_x = range_x[0] + obstacle_size[0], range_x[1] - obstacle_size[0]
        start_y, end_y = range_y[0] + 2, range_y[1] - 2

        for x in np.arange(start_x, end_x, x_step):
            for y in np.arange(start_y, end_y, y_step):
                obstacle = copy.deepcopy(self.params["objects"]["adjustable"])
                obstacle["name"] = f"Obstacle {len(additions)}"

                random_yaw = random.uniform(-np.pi, np.pi)

                init_pos = [x, y, 0.5]
                init_ori = self.yaw_to_quaternion(random_yaw)

                obstacle["init_pos"] = init_pos
                obstacle["init_ori"] = init_ori

                if not self.is_obstacle_overlapping(init_pos, obstacle["size"], init_ori,
                                                    additions, excluded_poses, margin=0.):
                    additions.append(obstacle)

        return additions

    def create_walls(self, thickness=0.01, height=0.5):
        range_x = self.params["range_x"]
        range_y = self.params["range_y"]

        def new_wall(name, size, init_pos, init_ori):
            wall = copy.deepcopy(self.params["objects"]['wall'])
            wall["name"] = name
            wall["size"] = size
            wall["init_pos"] = init_pos
            wall["init_ori"] = self.yaw_to_quaternion(init_ori[-1])

            return wall

        walls = []

        walls.append(new_wall("l-demarcation-wall", [range_x[1]-range_x[0], thickness, height], [
                     (range_x[1]+range_x[0])/2, range_y[1]+thickness/2, 0], [0.0, 0.0, 0.0]))
        walls.append(new_wall("r-demarcation-wall", [range_x[1]-range_x[0], thickness, height], [
                     (range_x[1]+range_x[0])/2, range_y[0]-thickness/2, 0], [0.0, 0.0, 0.0]))
        walls.append(new_wall("f-demarcation-wall", [thickness, range_y[1]-range_y[0], height], [
                     range_x[0]-thickness/2, (range_y[1]+range_y[0])/2, 0], [0.0, 0.0, 0.0]))
        walls.append(new_wall("b-demarcation-wall", [thickness, range_y[1]-range_y[0], height], [
                     range_x[1]+thickness/2, (range_y[1]+range_y[0])/2, 0], [0.0, 0.0, 0.0]))

        return walls

    def run(self, use_replanner=True):
        df_state_tensor = self.torch_to_bytes(self.simulation.dof_state)
        rt_state_tensor = self.torch_to_bytes(self.simulation.root_state)
        rb_state_tensor = self.torch_to_bytes(self.simulation.rigid_body_state)

        bytes_action = self.controller.compute_action_tensor(
            df_state_tensor, rt_state_tensor, rb_state_tensor)
        action = self.bytes_to_torch(bytes_action, self.device)

        if torch.any(torch.isnan(action)):
            action = torch.zeros_like(action)

        self.check_goal_reached()
        if self.is_goal_reached:
            rospy.loginfo_throttle(1, 'The goal is reached, no action applied to the robot')

        self.simulation.apply_robot_cmd(action)
        self.simulation.step()

        if not use_replanner:
            return action, False

        replan = False if self.is_goal_reached else self.evaluate_push_action(action)
        return action, replan

    def destroy(self):
        self.simulation.stop_sim()

    def evaluate_push_action(self, action):
        robot_dof = self.get_robot_dofs()
        q_rob = np.array([robot_dof[0], robot_dof[2], robot_dof[4]])
        q_dot_rob = np.array([robot_dof[1], robot_dof[3], robot_dof[5]])

        action_array = action.cpu().numpy()
        if np.all(np.abs(q_dot_rob[:2]) < 1e-1):
            rospy.logwarn_throttle(2, "Velocities are too close to zero, watchdog active")
        #elif np.all(np.abs(q_dot_rob - action_array) > 0.75 * np.abs(action_array)):
        #    rospy.logwarn_throttle(2, "Desired velocity is more than 75% away, watchdog active")
        elif np.all(np.abs(q_rob) < 1e-1) and np.any(np.abs(q_dot_rob) > 1e-1):
            rospy.logwarn_throttle(2, "Robot is slipping, watchdog active")
        else:
            self.replan_watchdog = time.time()

        if time.time() - self.replan_watchdog >= self.replan_timing:
            rospy.logwarn_throttle(1, "Replanning is necessary, evaluate obstacle mass and update estimation.")
            self.replan_watchdog = time.time()
            self.update_estimation()
            return True

        return False

    def update_estimation(self):
        closest_index = self.get_closest_obstacle_index()
        self.set_obstacle_mass(torch.tensor([closest_index]), torch.tensor([1000.0]))

    def update_objective(self, waypoints):
        torch_waypoints = torch.from_numpy(waypoints).to(self.device)
        bytes_waypoints = self.torch_to_bytes(torch_waypoints)

        self.controller.update_objective(bytes_waypoints)
        self._goal = waypoints[-1, :]

    def is_finished(self):
        if self.is_goal_reached:
            return True

        return False

    def check_goal_reached(self):
        if self._goal is None:
            return None

        rob_dof = self.get_robot_dofs()
        rob_pos = torch.tensor([rob_dof[0], rob_dof[2]], device=self.device)
        xy_goal = torch.tensor(self._goal[:2], device=self.device)

        if torch.linalg.norm(xy_goal - rob_pos) < self.pos_tolerance:
            self.is_goal_reached = True
        else:
            self.is_goal_reached = False

    def get_net_forces(self):
        net_contact_forces = torch.sum(torch.abs(torch.cat((self.simulation.net_contact_force[:, 0].unsqueeze(1), self.simulation.net_contact_force[:, 1].unsqueeze(1)), 1)), 1)
        number_of_bodies = int(net_contact_forces.size(dim=0) / self.simulation.num_envs)

        reshaped_contact_forces = net_contact_forces.reshape([self.simulation.num_envs, number_of_bodies])
        return torch.sum(reshaped_contact_forces, dim=1)[0].tolist()

    def get_robot_dofs(self):
        return self.simulation.dof_state[0].tolist()

    def get_actor_states(self):
        return self.simulation.env_cfg, self.simulation.root_state[0, :, :].cpu().numpy()

    def get_elapsed_time(self):
        return self.simulation._gym.get_elapsed_time(self.simulation.sim)

    def get_rollout_states(self):
        return self.bytes_to_torch(self.controller.get_states(), self.device)

    def get_rollout_best_state(self):
        return self.bytes_to_torch(self.controller.get_n_best_samples(), self.device)

    def get_closest_obstacle_index(self):
        _, actors_states = self.get_actor_states()
        robot_dof = self.get_robot_dofs()

        robot_x, robot_y = robot_dof[0], robot_dof[2]
        actors_states = torch.from_numpy(actors_states)

        obstacle_positions = actors_states[:, :2]
        robot_position = torch.tensor([robot_x, robot_y])

        distances = torch.norm(obstacle_positions - robot_position, dim=1)

        closest_obstacle_index = torch.argmin(distances).item()
        return closest_obstacle_index

    def set_obstacle_mass(self, obstacle_index, obstacle_mass):
        self.simulation.set_actor_mass_by_actor_index(obstacle_index, obstacle_mass)

        obstacle_index = self.torch_to_bytes(obstacle_index)
        obstacle_mass = self.torch_to_bytes(obstacle_mass)
        self.controller.set_actor_mass_by_actor_index(obstacle_index, obstacle_mass)

    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(
            *position), gymapi.Vec3(*target))

    @staticmethod
    def yaw_to_quaternion(yaw):
        return quaternion_from_euler(0., 0., yaw).tolist()

    @staticmethod
    def quaternion_to_yaw(quaternion):
        return euler_from_quaternion(quaternion)[-1]

    @staticmethod
    def is_obstacle_overlapping(new_position, new_size, new_orientation, obstacles, excluded_poses, margin=0.1):
        new_polygon = Polygon([
            (new_position[0] - new_size[0] / 2,
             new_position[1] - new_size[1] / 2),
            (new_position[0] + new_size[0] / 2,
             new_position[1] - new_size[1] / 2),
            (new_position[0] + new_size[0] / 2,
             new_position[1] + new_size[1] / 2),
            (new_position[0] - new_size[0] / 2,
             new_position[1] + new_size[1] / 2)
        ])

        yaw = np.rad2deg(SimulateWorld.quaternion_to_yaw(new_orientation))
        new_polygon = rotate(new_polygon, yaw)
        new_polygon = shapely.buffer(
            new_polygon, margin, cap_style='flat', join_style='bevel')

        all_excluded_poses = list(obstacles) + list(excluded_poses)
        for excluded_pose in all_excluded_poses:
            obs_pos = excluded_pose['init_pos']
            obs_yaw = np.rad2deg(
                SimulateWorld.quaternion_to_yaw(excluded_pose['init_ori']))
            obs_size = excluded_pose['size']

            corners = Polygon([
                (obs_pos[0] - obs_size[0] / 2, obs_pos[1] - obs_size[1] / 2),
                (obs_pos[0] + obs_size[0] / 2, obs_pos[1] - obs_size[1] / 2),
                (obs_pos[0] + obs_size[0] / 2, obs_pos[1] + obs_size[1] / 2),
                (obs_pos[0] - obs_size[0] / 2, obs_pos[1] + obs_size[1] / 2)
            ])

            obstacle = Polygon(corners)
            obstacle = rotate(obstacle, obs_yaw)
            obstacle = shapely.buffer(
                obstacle, margin, cap_style='flat', join_style='bevel')

            if new_polygon.intersects(obstacle):
                return True

        return False

    @staticmethod
    def torch_to_bytes(torch_tensor):
        buff = io.BytesIO()
        torch.save(torch_tensor, buff)
        buff.seek(0)
        return buff.read()

    @staticmethod
    def bytes_to_torch(buffer, device):
        buff = io.BytesIO(buffer)
        return torch.load(buff, map_location=device)

    @property
    def goal(self):
        return self._goal

    @property
    def mode(self):
        return self._mode
