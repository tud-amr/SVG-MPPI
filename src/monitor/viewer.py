from control.mppi_isaac.mppiisaac.planner.isaacgym_wrapper import IsaacGymWrapper, ActorWrapper    # type: ignore
from control.mppi_isaac.mppiisaac.utils.config_store import ExampleConfig                          # type: ignore

import rospy
import roslib
import torch
import yaml


from functools import partial
from scipy.spatial.transform import Rotation
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped

from isaacgym import gymapi


class Viewer:

    PKG_PATH = roslib.packages.get_pkg_dir("semantic_namo")

    def __init__(self, params, config, simulation):
        self.simulation = simulation

        self.params = params
        self.config = config

        self.obstacle_states = []

        self.robot_prev_msg = None
        self.robot_q_dot = torch.zeros(3)
        self.robot_q = torch.zeros(3)

    @classmethod
    def build(cls, config: ExampleConfig, layout: str, robot_name: str):
        viewer = cls.create(config, layout)
        viewer.configure(robot_name)
        return viewer

    @classmethod
    def create(cls, config, layout):
        simulation = IsaacGymWrapper(
            config["isaacgym"],
            init_positions=config["initial_actor_positions"],
            actors=config["actors"],
            num_envs=1,
            viewer=True,
            device=config["mppi"].device,
        )

        base_config_file_path = f'{cls.PKG_PATH}/config/worlds/base.yaml'
        with open(base_config_file_path, 'r') as stream:
            base_config =  yaml.safe_load(stream)

        world_config_file_path = f'{cls.PKG_PATH}/config/worlds/{layout}.yaml'
        with open(world_config_file_path, 'r') as stream:
            world_config =  yaml.safe_load(stream)

        params = {**base_config, **world_config}

        return cls(params, config, simulation)
    
    def configure(self, robot_name):
        additions = self.create_additions()
        self.simulation.add_to_envs(additions)

        rospy.Subscriber(f'/vicon/{robot_name}', PoseStamped, self._cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseStamped, timeout=10)

        cam_pos = self.params["camera"]["pos"]
        cam_tar = self.params["camera"]["tar"]
        self.set_viewer(self.simulation._gym, self.simulation.viewer, cam_pos, cam_tar)

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

    def run(self):
        for idx, obstacle_state in enumerate(self.obstacle_states):
            obs_state = torch.tensor([*obstacle_state, 0., 0., 0., 0., 0., 0.], device=self.config["mppi"]["device"])
            self.simulation.set_root_state_tensor_by_actor_idx(obs_state, idx + 1)

        self.simulation.reset_robot_state(self.robot_q, self.robot_q_dot)
        self.simulation.step()

    def destroy(self):
        self.simulation.stop_sim()

    def _cb_robot_state(self, msg):
        curr_pos = torch.tensor([msg.pose.position.x, msg.pose.position.y])
        curr_ori = msg.pose.orientation

        _, _, curr_yaw = euler_from_quaternion([curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w])

        self.robot_q = torch.tensor([curr_pos[0], curr_pos[1], curr_yaw])

        if self.robot_prev_msg is not None:
            prev_pos = torch.tensor([self.robot_prev_msg.pose.position.x, self.robot_prev_msg.pose.position.y])
            prev_ori = self.robot_prev_msg.pose.orientation

            _, _, prev_yaw = euler_from_quaternion([prev_ori.x, prev_ori.y, prev_ori.z, prev_ori.w])

            delta_t = msg.header.stamp.to_sec() - self.robot_prev_msg.header.stamp.to_sec()

            linear_velocity = (curr_pos - prev_pos) / delta_t
            angular_velocity = (curr_yaw - prev_yaw) / delta_t

            self.robot_q_dot = torch.tensor([linear_velocity[0], linear_velocity[1], angular_velocity])
            
        self.robot_prev_msg = msg

    def _cb_obstacle_state(self, idx, msg):
        pos, ori = msg.pose.position, msg.pose.orientation
        _, _, obs_yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.obstacle_states[idx] = [pos.x, pos.y, 0., *self.yaw_to_quaternion(obs_yaw)]

    @staticmethod
    def yaw_to_quaternion(yaw):
        return quaternion_from_euler(0., 0., yaw)
    
    @staticmethod
    def set_viewer(gym, viewer, position, target):
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(*position), gymapi.Vec3(*target))
