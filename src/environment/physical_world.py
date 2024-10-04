from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import ActorWrapper     # type: ignore
from control.mppi_isaac.mppiisaac.planner.mppi_isaac import MPPIisaacPlanner       # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig          # type: ignore

from motion import Dingo
from scheduler import Objective

import io
import rospy
import roslib
import torch
import time
import yaml
import numpy as np

from functools import partial
from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped


class PhysicalWorld:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    MSG_TIMEOUT = 3
    
    def __init__(self, params, config, controller, device):        
        self.controller = controller
        self.env_config = None

        self.params = params
        self.config = config
        self.device = device

        self.pos_tolerance = params['controller']['pos_tolerance']
        self.yaw_tolerance = params['controller']['yaw_tolerance']
        self.vel_tolerance = params['controller']['vel_tolerance']
        self.replan_timing = params['controller']['replan_timing']

        self.robot = None

        self.obstacle_states = []

        self.robot_prev_msg = None
        self.robot_q_dot = torch.zeros(3)
        self.robot_R = torch.zeros(2, 2)
        self.robot_q = torch.zeros(3)

        self._goal = None
        self._mode = None

        self.is_goal_reached = False
        self.replan_watchdog = None

    @classmethod
    def build(cls, config: ExampleConfig, layout: str, robot_name: str):
        world = PhysicalWorld.create(config, layout)
        world.configure(robot_name)
        return world
    
    @classmethod
    def create(cls, config, layout):
        actors=[]
        for actor_name in config["actors"]:
            with open(f'{cls.PKG_PATH}/config/actors/{actor_name}.yaml') as f:
                actors.append(ActorWrapper(**yaml.load(f, Loader=yaml.SafeLoader)))

        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config =  yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config =  yaml.safe_load(stream)

        params = {**base_config, **world_config}
        
        objective = Objective(config["mppi"].u_min, config["mppi"].u_max, config["mppi"].device)

        controller = MPPIisaacPlanner(config, objective)
        return cls(params, config, controller, config["mppi"].device)

    def configure(self, robot_name):
        additions = self.create_additions()
        self.controller.add_to_env(additions)
        
        rospy.Subscriber(f'/vicon/{robot_name}', PoseStamped, self._cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseStamped, timeout=10)
        
        robot_q = self.robot_q.cpu()

        self.update_objective(np.array([[robot_q[0], robot_q[1]]]))
        self.robot = Dingo(robot_name)

        self.replan_watchdog = time.time()

    def create_additions(self):
        additions =[]

        if self.params.get("environment", None):
            if self.params["environment"].get("obstacles", None):
                for idx, obstacle in enumerate(self.params["environment"]["obstacles"]):
                    obs_type = next(iter(obstacle))
                    obs_args = self.params["objects"][obs_type]

                    obstacle = {**obs_args, **obstacle[obs_type]}
                    topic_name = obstacle.get("topic_name", None)

                    self.obstacle_states.append([*obstacle["init_pos"], 0., 0., 0., 1.])
                    
                    if topic_name is not None:
                        rospy.Subscriber(f'/vicon/{topic_name}', PoseStamped, partial(self._cb_obstacle_state, idx), queue_size=1)
                        rospy.wait_for_message(f'/vicon/{topic_name}', PoseStamped, timeout=10)

                    additions.append(obstacle)

            if self.params["environment"].get("demarcation", None):
                for wall in self.params["environment"]["demarcation"]:
                    obs_type = next(iter(wall))
                    obs_args = self.params["objects"][obs_type]

                    obstacle = {**obs_args, **wall[obs_type]}

                    rot = Rotation.from_euler('xyz', obstacle["init_ori"], degrees=True).as_quat()
                    obstacle["init_ori"] = list(rot)

                    self.obstacle_states.append([*obstacle["init_pos"], *obstacle["init_ori"]])

                    additions.append(obstacle)

        return additions

    def run(self, use_replanner=True):
        if self.robot_prev_msg is not None:
            obstacle_state_tensor = torch.tensor(self.obstacle_states)
            obstacle_state_tensor = self.add_zero_columns(obstacle_state_tensor)

            action = self.controller.compute_action(self.robot_q, self.robot_q_dot, obst_tensor=obstacle_state_tensor)
            action[:2] = torch.matmul(self.robot_R.T, action[:2])

            if not self.is_goal_reached:
                self.robot.move(*action)
            else:
                rospy.loginfo_throttle(1, "The goal is reached, no action applied to the robot.")

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
        if np.all(np.abs(q_dot_rob) < 1e-1):
            rospy.logwarn_throttle(2, "Velocities are too close to zero, watchdog active")
        elif np.all(np.abs(q_dot_rob - action_array) > 0.5 * np.abs(action_array)):
            rospy.logwarn_throttle(2, "Desired velocity is more than 50% away, watchdog active")
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
        torch_waypoints  = torch.from_numpy(waypoints).float()
        torch_waypoints = torch_waypoints.to(self.device)

        self.controller.update_objective(torch_waypoints)
        self._goal = waypoints[-1, :]

    def is_finished(self):
        if self.is_goal_reached:
            if self.robot_q_dot[0] < self.vel_tolerance and self.robot_q_dot[1] < self.vel_tolerance:
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

    def _cb_robot_state(self, msg):
        curr_pos = torch.tensor([msg.pose.position.x, msg.pose.position.y])
        curr_ori = msg.pose.orientation

        _, _, curr_yaw = euler_from_quaternion([curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w])

        self.robot_q = torch.tensor([curr_pos[0], curr_pos[1], curr_yaw], dtype=torch.double, device=self.device)

        if self.robot_prev_msg is not None:
            prev_pos = torch.tensor([self.robot_prev_msg.pose.position.x, self.robot_prev_msg.pose.position.y])
            prev_ori = self.robot_prev_msg.pose.orientation

            _, _, prev_yaw = euler_from_quaternion([prev_ori.x, prev_ori.y, prev_ori.z, prev_ori.w])

            delta_t = msg.header.stamp.to_sec() - self.robot_prev_msg.header.stamp.to_sec()

            linear_velocity = (curr_pos - prev_pos) / delta_t
            angular_velocity = (curr_yaw - prev_yaw) / delta_t

            cos_yaw = torch.cos(torch.tensor([curr_yaw]))
            sin_yaw = torch.sin(torch.tensor([curr_yaw]))
            self.robot_R = torch.tensor([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], device=self.device)
            
            self.robot_q_dot = torch.tensor([linear_velocity[0], linear_velocity[1], angular_velocity], dtype=torch.double,device=self.device)
            
            self.check_goal_reached()

        self.robot_prev_msg = msg

    def _cb_obstacle_state(self, idx, msg):
        pos, ori = msg.pose.position, msg.pose.orientation
        _, _, obs_yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.obstacle_states[idx] = [pos.x, pos.y, 0., *self.yaw_to_quaternion(obs_yaw)]

    def get_robot_dofs(self):
        q, q_dot = self.robot_q, self.robot_q_dot
        return torch.tensor([q[0], q_dot[0], q[1], q_dot[1], q[2], q_dot[2]])

    def get_actor_states(self):
        rob_state = torch.tensor([self.robot_q[0], self.robot_q[1], 0., 0., 0., 0., 0.])
        obs_state = torch.tensor(self.obstacle_states) if self.obstacle_states else torch.empty((0, 7))

        states = torch.empty((len(self.obstacle_states) + 1, 7))
        states[0] = rob_state

        if obs_state.numel() > 0:
            states[1:] = obs_state

        return self.controller.sim.env_cfg, states

    def get_rollout_states(self):
        return self.bytes_to_torch(self.controller.get_states())

    def get_rollout_best_state(self):
        return self.bytes_to_torch(self.controller.get_n_best_samples())

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
        obstacle_index = self.torch_to_bytes(obstacle_index)
        obstacle_mass = self.torch_to_bytes(obstacle_mass)
        
        self.controller.set_actor_mass(obstacle_index, obstacle_mass)

    @staticmethod
    def add_zero_columns(tensor, total_columns=13):
        current_columns = tensor.size(1)
        if current_columns < total_columns:
            num_zero_columns = total_columns - current_columns
            zero_columns = torch.zeros((tensor.size(0), num_zero_columns), dtype=tensor.dtype)
            tensor = torch.cat((tensor, zero_columns), dim=1)
        return tensor

    @staticmethod
    def yaw_to_quaternion(yaw):
        return quaternion_from_euler(0., 0., yaw)

    @staticmethod
    def bytes_to_torch(buffer):
        buff = io.BytesIO(buffer)
        return torch.load(buff)

    @property
    def goal(self):
        return self._goal
    
    @property
    def mode(self):
        return self._mode
    