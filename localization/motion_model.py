
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry

class MotionModel(Node):

    def __init__(self):
        super().__init__("motion_model")

        # Declare noise parameters (to tune in YAML file)
        self.sigma_x = (
            self.declare_parameter("sigma_x", 1.0)
            .get_parameter_value()
            .double_value
        )
        self.sigma_y = (
            self.declare_parameter("sigma_y", 0.5)
            .get_parameter_value()
            .double_value
        )
        self.sigma_theta = (
            self.declare_parameter("sigma_theta", 0.7)
            .get_parameter_value()
            .double_value
        )

        # Subscribe to odometry
        self.latest_odom = None
        self.sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.get_logger().info("Motion model initialized.")

    # Store the most recent odometry message
    def odom_callback(self, msg):
        self.latest_odom = msg

    # Main motion model used by the particle filter
    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta] in the BODY frame

        returns:
            particles: An updated matrix of the
                same size
        """

        if particles is None or len(particles) == 0:
            return particles

        # Extract odometry increment
        dx, dy, dtheta = odometry

        # Add Gaussian noise to odometry increment
        dx_noisy = dx + np.random.normal(0, self.sigma_x, size=len(particles))
        dy_noisy = dy + np.random.normal(0, self.sigma_y, size=len(particles))
        dtheta_noisy = dtheta + np.random.normal(0, self.sigma_theta, size=len(particles))

        # Apply motion model to each particle
        x = particles[:, 0]
        y = particles[:, 1]
        theta = particles[:, 2]

        # Rotation matrix for each particle
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Transform body-frame motion into world frame
        x_new = x + cos_t * dx_noisy - sin_t * dy_noisy
        y_new = y + sin_t * dx_noisy + cos_t * dy_noisy
        theta_new = theta + dtheta_noisy

        # Normalize angle
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        # Return updated particle set
        return np.vstack((x_new, y_new, theta_new)).T
