from isaacgym import gymapi

import pickle
import rospy
import torch

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

from tf.transformations import euler_from_quaternion

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from puma_motor_msgs.msg import MultiFeedback



class Monitor:
    def __init__(self, is_simulation):
        self._is_simulation = is_simulation
        self.active_monitor = False

        self._prev_msg_robot_state = None
        self._prev_elapsed_cb_time = None

        self.rate_loop = {"t": [], "duration": []}
        self.rob_state = {"t": [], "x": [], "y": [], "yaw": []}
        self.rob_twist = {"t": [], "x_d": [], "y_d": [], "yaw_d": []}

        self.joint_pos = {"t": [], "fl": [], "fr": [], "rl": [], "rr": []}
        self.joint_vel = {"t": [], "fl": [], "fr": [], "rl": [], "rr": []}
        self.joint_eff = {"t": [], "fl": [], "fr": [], "rl": [], "rr": []}

        self.wheel_cur = {"t": [], "fl": [], "fr": [], "rl": [], "rr": []}
        self.wheel_vol = {"t": [], "fl": [], "fr": [], "rl": [], "rr": []}

        self.twist_cmd = {"t": [], "lin_x": [], "lin_y": [],"ang_z": []}

    def configure_physical_monitor(self, robot_name):
        rospy.Subscriber(f'/vicon/{robot_name}', PoseStamped, self.cb_robot_state, queue_size=1,)
        rospy.wait_for_message(f'/vicon/{robot_name}', PoseStamped, timeout=10)

        rospy.Subscriber(f'/{robot_name}/feedback', MultiFeedback, self.cb_robot_feedback)
        rospy.wait_for_message(f'/{robot_name}/feedback', MultiFeedback, timeout=10)

        rospy.Subscriber(f'/{robot_name}/joint_states', JointState, self.cb_joint_states)
        rospy.wait_for_message(f'/{robot_name}/joint_states', JointState, timeout=10)

        rospy.Subscriber(f'/{robot_name}/cmd_vel', Twist, self.cb_cmd_vel)

    def configure_simulate_monitor(cls, sim):
        for actor in range(1, len(sim.env_cfg)):
            actor_wrapper = sim.env_cfg[actor]

            mass = cls.get_sim_actor_mass(sim, actor)
            name = cls.get_sim_actor_name(sim, actor)

            rospy.loginfo(f'Actor: {actor}, name: {name}, mass: {mass}, size: {actor_wrapper.size}, fixed: {actor_wrapper.fixed}')        

    def start_monitoring(self):
        self._prev_elapsed_cb_time = float(0)
        self.start_time = rospy.get_time()
        self.active_monitor = True

    def stop_monitoring(self):
        self.active_monitor = False

    def step_monitoring(self):
        if self.active_monitor:
            elapsed_time = rospy.get_time() - self.start_time
            duration = elapsed_time - self._prev_elapsed_cb_time

            self.rate_loop["t"].append(elapsed_time)
            self.rate_loop["duration"].append(duration)

            self._prev_elapsed_cb_time = elapsed_time

    def cb_simulate(self, sim, cmd_vel):
        if self.active_monitor:
            elapsed_time = rospy.get_time() - self.start_time

            state = self.get_sim_robot_state(sim)
            state_d = self.get_sim_robot_state_d(sim)

            self.rob_state["t"].append(elapsed_time)
            self.rob_state["x"].append(state[0])
            self.rob_state["y"].append(state[1])
            self.rob_state["yaw"].append(state[2])

            self.rob_twist["t"].append(elapsed_time)
            self.rob_twist["x_d"].append(state_d[0])
            self.rob_twist["y_d"].append(state_d[1])
            self.rob_twist["yaw_d"].append(state_d[2])

            self.twist_cmd["t"].append(elapsed_time)
            self.twist_cmd["lin_x"].append(cmd_vel[0])
            self.twist_cmd["lin_y"].append(cmd_vel[1])
            self.twist_cmd["ang_z"].append(cmd_vel[2])

    def cb_robot_state(self, msg):
        if self.active_monitor:
            elapsed_time = rospy.get_time() - self.start_time

            curr_pos = np.array([msg.pose.position.x, msg.pose.position.y])
            curr_ori = msg.pose.orientation

            _, _, curr_yaw = euler_from_quaternion([curr_ori.x, curr_ori.y, curr_ori.z, curr_ori.w])

            if self._prev_msg_robot_state is not None:
                prev_pos = np.array([self._prev_msg_robot_state.pose.position.x, self._prev_msg_robot_state.pose.position.y])
                prev_ori = self._prev_msg_robot_state.pose.orientation

                _, _, prev_yaw = euler_from_quaternion([prev_ori.x, prev_ori.y, prev_ori.z, prev_ori.w])

                delta_t = msg.header.stamp.to_sec() - self._prev_msg_robot_state.header.stamp.to_sec()

                lin_vel = (curr_pos - prev_pos) / delta_t
                ang_vel = (curr_yaw - prev_yaw) / delta_t

                self.rob_state["t"].append(elapsed_time)
                self.rob_state["x"].append(curr_pos[0])
                self.rob_state["y"].append(curr_pos[1])
                self.rob_state["yaw"].append(curr_yaw)

                self.rob_twist["t"].append(elapsed_time)
                self.rob_twist["x_d"].append(lin_vel[0])
                self.rob_twist["y_d"].append(lin_vel[1])
                self.rob_twist["yaw_d"].append(ang_vel)

        self._prev_msg_robot_state = msg

    def cb_robot_feedback(self, msg):
        if self.active_monitor:
            elapsed_time = rospy.get_time() - self.start_time

            self.wheel_cur["t"].append(elapsed_time)
            self.wheel_cur["fl"].append(msg.drivers_feedback[0].current)
            self.wheel_cur["fr"].append(msg.drivers_feedback[1].current)
            self.wheel_cur["rl"].append(msg.drivers_feedback[2].current)
            self.wheel_cur["rr"].append(msg.drivers_feedback[3].current)

            self.wheel_vol["t"].append(elapsed_time)
            self.wheel_vol["fl"].append(msg.drivers_feedback[0].duty_cycle)
            self.wheel_vol["fr"].append(msg.drivers_feedback[1].duty_cycle)
            self.wheel_vol["rl"].append(msg.drivers_feedback[2].duty_cycle)
            self.wheel_vol["rr"].append(msg.drivers_feedback[3].duty_cycle)

        self.twist_cmd = {"t": [], "lin_x": [], "lin_y": [],"ang_z": []}

    def cb_joint_states(self, msg):
        if self.active_monitor:
            elapsed_time = rospy.get_time() - self.start_time

            self.joint_pos["t"].append(elapsed_time)
            self.joint_pos["fl"].append(msg.position[0])
            self.joint_pos["fr"].append(msg.position[1])
            self.joint_pos["rl"].append(msg.position[2])
            self.joint_pos["rr"].append(msg.position[3])

            self.joint_vel["t"].append(elapsed_time)
            self.joint_vel["fl"].append(msg.velocity[0])
            self.joint_vel["fr"].append(msg.velocity[1])
            self.joint_vel["rl"].append(msg.velocity[2])
            self.joint_vel["rr"].append(msg.velocity[3])

            self.joint_eff["t"].append(elapsed_time)
            self.joint_eff["fl"].append(msg.effort[0])
            self.joint_eff["fr"].append(msg.effort[1])
            self.joint_eff["rl"].append(msg.effort[2])
            self.joint_eff["rr"].append(msg.effort[3])

    def cb_cmd_vel(self, msg):
        if self.active_monitor:
            elapsed_time = rospy.get_time() - self.start_time

            self.twist_cmd["t"].append(elapsed_time)
            self.twist_cmd["lin_x"].append(msg.linear.x)
            self.twist_cmd["lin_y"].append(msg.linear.y)
            self.twist_cmd["ang_z"].append(msg.angular.z)

    def request_save(self):
        root = tk.Tk()
        root.withdraw()

        response = messagebox.askyesno("Save the data", "Do you want to save the dataframe?")
        root.destroy()

        rospy.loginfo(f"Save dataframe is set to {response}")

        if response:
            root = tk.Tk()
            root.withdraw()
            
            file_path = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
            root.destroy()

            if not file_path:
                rospy.loginfo("Save operation canceled")
            else:
                rospy.loginfo(f"Saving data to: {file_path}")
                with open(file_path, 'wb') as f:
                    pickle.dump(self.data, f)

    @staticmethod
    def get_sim_robot_state(sim):
        state = torch.cat((sim.dof_state[:, 0].unsqueeze(1), sim.dof_state[:, 2].unsqueeze(1), sim.dof_state[:, 4].unsqueeze(1)), 1)[0].cpu().numpy()
        return state
    
    @staticmethod
    def get_sim_robot_state_d(sim):
        state_d = torch.cat((sim.dof_state[:, 1].unsqueeze(1), sim.dof_state[:, 3].unsqueeze(1), sim.dof_state[:, 5].unsqueeze(1)), 1)[0].cpu().numpy()
        return state_d

    @staticmethod
    def get_sim_actor_mass(sim, actor):
        rigid_body_property = sim._gym.get_actor_rigid_body_properties(sim.envs[0], actor)[0]
        return rigid_body_property.mass

    @staticmethod
    def get_sim_actor_name(sim, actor):
        return sim._gym.get_actor_name(sim.envs[0], actor)

    @property
    def rob_energy(self):
        rob_power = np.abs(np.multiply(self.wheel_cur['fl'], self.wheel_vol['fl'])) + \
                    np.abs(np.multiply(self.wheel_cur['fr'], self.wheel_vol['fr'])) + \
                    np.abs(np.multiply(self.wheel_cur['rl'], self.wheel_vol['rl'])) + \
                    np.abs(np.multiply(self.wheel_cur['rr'], self.wheel_vol['rr']))

        rob_current = np.add(np.add(np.add(self.wheel_cur['fl'], self.wheel_cur['fr']), self.wheel_cur['rl']), self.wheel_cur['rr']) 
        rob_voltage = np.add(np.add(np.add(self.wheel_vol['fl'], self.wheel_vol['fr']), self.wheel_vol['rl']), self.wheel_vol['rr']) 

        cumulative_power = np.cumsum(rob_power)
        cumulative_current = np.cumsum(rob_current)
        cumulative_voltage = np.cumsum(rob_voltage)

        return {'t': self.wheel_cur['t'], 
                'power': rob_power.tolist(),
                'current': rob_current.tolist(), 
                'voltage': rob_voltage.tolist(),
                'cumsum_power': cumulative_power.tolist(),
                'cumsum_current': cumulative_current.tolist(),
                'cumsum_voltage': cumulative_voltage.tolist()}

    @property
    def data(self):
        return {
            "rate_loop": pd.DataFrame(self.rate_loop),
            "robot_state": pd.DataFrame(self.rob_state),
            "robot_twist": pd.DataFrame(self.rob_twist),
            "joint_position": pd.DataFrame(self.joint_pos),
            "joint_velocity": pd.DataFrame(self.joint_vel),
            "joint_effort": pd.DataFrame(self.joint_eff),
            "wheel_current": pd.DataFrame(self.wheel_cur),
            "wheel_voltage": pd.DataFrame(self.wheel_vol),
            "twist_command": pd.DataFrame(self.twist_cmd),
            "robot_energy": pd.DataFrame(self.rob_energy)}
