import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, Pose


class MotionModel(Node):

    def __init__(self):
        super().__init__("motion_model")

        # Declare noise parameters
        self.sigma_x = (
            self.declare_parameter("sigma_x", 1.0)
            .get_parameter_value()
            .double_value
        )
        self.sigma_y = (
            self.declare_parameter("sigma_y", 1.0)
            .get_parameter_value()
            .double_value
        )
        self.sigma_theta = (
            self.declare_parameter("sigma_theta", 1.0)
            .get_parameter_value()
            .double_value
        )

        # NEW: Deterministic flag
        self.deterministic = (
            self.declare_parameter("deterministic", False)
            .get_parameter_value()
            .bool_value
        )

        # Subscribe to odometry
        self.latest_odom = None
        self.sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        # Publisher for pose output
        self.pose_pub = self.create_publisher(
            PoseArray,
            "/motion_model_pose",
            10
        )

        # Time tracking
        self.prev_time = self.get_clock().now()

        self.get_logger().info(
            f"Motion model initialized. Deterministic = {self.deterministic}"
        )

    def odom_callback(self, msg):
        self.latest_odom = msg

    def evaluate(self, particles):

        if particles is None or len(particles) == 0:
            return particles

        if self.latest_odom is None:
            return particles  

        odom = self.latest_odom

        # Extract velocities
        vx = odom.twist.twist.linear.x
        vy = odom.twist.twist.linear.y
        wz = odom.twist.twist.angular.z

        # Compute dt
        current_time = self.get_clock().now()
        dt = (current_time - self.prev_time).nanoseconds * 1e-9
        self.prev_time = current_time

        dx = vx * dt
        dy = vy * dt
        dtheta = wz * dt

        # Deterministic vs. stochastic motion model
        if self.deterministic:
            dx_noisy = np.full(len(particles), dx)
            dy_noisy = np.full(len(particles), dy)
            dtheta_noisy = np.full(len(particles), dtheta)
        else:
            dx_noisy = dx + np.random.normal(0, self.sigma_x, size=len(particles))
            dy_noisy = dy + np.random.normal(0, self.sigma_y, size=len(particles))
            dtheta_noisy = dtheta + np.random.normal(0, self.sigma_theta, size=len(particles))

        # Apply motion model
        x = particles[:, 0]
        y = particles[:, 1]
        theta = particles[:, 2]

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x_new = x + cos_t * dx_noisy - sin_t * dy_noisy
        y_new = y + sin_t * dx_noisy + cos_t * dy_noisy
        theta_new = theta + dtheta_noisy

        # Normalize angle
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        updated_particles = np.vstack((x_new, y_new, theta_new)).T

        # Publish updated particle poses
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        for p in updated_particles:
            pose = Pose()
            pose.position.x = float(p[0])
            pose.position.y = float(p[1])
            pose.position.z = 0.0

            # Convert theta → quaternion
            qz = np.sin(p[2] / 2.0)
            qw = np.cos(p[2] / 2.0)

            pose.orientation.z = float(qz)
            pose.orientation.w = float(qw)

            pose_array.poses.append(pose)

        self.pose_pub.publish(pose_array)

        # Echo to console
        self.get_logger().info(
            f"Published {len(updated_particles)} poses. Deterministic={self.deterministic}. Example: {updated_particles[0]}"
        )

        return updated_particles


