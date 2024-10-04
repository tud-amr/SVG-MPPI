
import math
import rospy

from geometry_msgs.msg import Twist


class Dingo:

    MAX_VEL_LIN = 1.0
    MAX_VEL_ANG = 1.0

    def __init__(self, robot_name):
        self._vel_pub = rospy.Publisher(f'/{robot_name}/cmd_vel', Twist, queue_size=1)

    def move(self, vel_lin_x=0., vel_lin_y=0., vel_ang_z=0.):
        vel_msg = Twist()

        if abs(vel_lin_x) > self.MAX_VEL_LIN:
            rospy.logwarn(f"Velocity command linear x is larger than maximum: {vel_lin_x} [max: {self.MAX_VEL_LIN}]")
            vel_lin_x = math.copysign(self.MAX_VEL_LIN, vel_lin_x)

        if abs(vel_lin_y) > self.MAX_VEL_LIN:
            rospy.logwarn(f"Velocity command linear y is larger than maximum: {vel_lin_y} [max: {self.MAX_VEL_LIN}]")
            vel_lin_y = math.copysign(self.MAX_VEL_LIN, vel_lin_y)

        if abs(vel_ang_z) > self.MAX_VEL_ANG:
            rospy.logwarn(f"Velocity command angular z is larger than maximum: {vel_ang_z} [max: {self.MAX_VEL_ANG}]")
            vel_ang_z = math.copysign(self.MAX_VEL_ANG, vel_ang_z)

        vel_msg.linear.x = vel_lin_x
        vel_msg.linear.y = vel_lin_y
        vel_msg.angular.z = vel_ang_z

        self._vel_pub.publish(vel_msg)
