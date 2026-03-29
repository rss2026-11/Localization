#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np

class OdomNoiseNode(Node):
    def __init__(self):
        super().__init__('odom_noise')

        # Declare parameters for noise (Gaussian std dev)
        self.declare_parameter('sigma_v', 0.5)
        self.declare_parameter('sigma_y', 0.0) # By default, 0 y-slip
        self.declare_parameter('sigma_w', 0.5)

        # # Extras for the assignment: bias and dropouts
        # self.declare_parameter('bias_v', 0.0)
        # self.declare_parameter('bias_y', 0.0)
        # self.declare_parameter('bias_w', 0.0)
        # self.declare_parameter('dropout_prob', 0.0)

        self.sigma_v = self.get_parameter('sigma_v').value
        self.sigma_y = self.get_parameter('sigma_y').value
        self.sigma_w = self.get_parameter('sigma_w').value
        # self.bias_v = self.get_parameter('bias_v').value
        # self.bias_y = self.get_parameter('bias_y').value
        # self.bias_w = self.get_parameter('bias_w').value
        # self.dropout_prob = self.get_parameter('dropout_prob').value

        self.sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.pub = self.create_publisher(
            Odometry,
            '/noisy_odom',
            10
        )

        self.get_logger().info(
            f"OdomNoiseNode started. sigma_v={self.sigma_v}, sigma_y={self.sigma_y}, sigma_w={self.sigma_w}"
        )

    def odom_callback(self, msg):
        # We can directly modify the incoming message as it's not used elsewhere in this node
        noisy_msg = msg

        # Simulate a completely dropped sensor reading (speed reports as 0)
        if np.random.rand() < self.dropout_prob:
            noisy_msg.twist.twist.linear.x = 0.0
            noisy_msg.twist.twist.linear.y = 0.0
            noisy_msg.twist.twist.angular.z = 0.0
        else:
            # Add Gaussian noise + bias
            noisy_msg.twist.twist.linear.x += np.random.normal(self.bias_v, self.sigma_v)
            noisy_msg.twist.twist.linear.y += np.random.normal(self.bias_y, self.sigma_y)
            noisy_msg.twist.twist.angular.z += np.random.normal(self.bias_w, self.sigma_w)

        self.pub.publish(noisy_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OdomNoiseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
