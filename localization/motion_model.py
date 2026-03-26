
import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        node.declare_parameter('deterministic', False)
        self.deterministic = node.get_parameter('deterministic').get_parameter_value().bool_value

        # Noise parameters — start small, tune later
        node.declare_parameter('sigma_x', 0.05)
        node.declare_parameter('sigma_y', 0.05)
        node.declare_parameter('sigma_theta', 0.02)

        self.sigma_x = node.get_parameter('sigma_x').get_parameter_value().double_value
        self.sigma_y = node.get_parameter('sigma_y').get_parameter_value().double_value
        self.sigma_theta = node.get_parameter('sigma_theta').get_parameter_value().double_value

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO

        dx, dy, dtheta = odometry
        n = particles.shape[0]

        # Add noise to odometry if not deterministic
        # if not self.deterministic:
        #     dx = dx + np.random.normal(0, self.sigma_x, n)
        #     dy = dy + np.random.normal(0, self.sigma_y, n)
        #     dtheta = dtheta + np.random.normal(0, self.sigma_theta, n)

        if not self.deterministic:
            n = particles.shape[0]
            dx = dx + np.random.normal(0, abs(dx) * self.sigma_x + 0.001, n)
            dy = dy + np.random.normal(0, abs(dy) * self.sigma_y + 0.001, n)
            dtheta = dtheta + np.random.normal(0, abs(dtheta) * self.sigma_theta + 0.0005, n)

        # Get current headings
        thetas = particles[:, 2]
        cos_t = np.cos(thetas)
        sin_t = np.sin(thetas)

        # Rotate body-frame odometry into world frame and apply
        particles[:, 0] += dx * cos_t - dy * sin_t
        particles[:, 1] += dx * sin_t + dy * cos_t
        particles[:, 2] += dtheta

        return particles

        ###################################