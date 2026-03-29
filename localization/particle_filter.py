from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TransformStamped
from tf_transformations import euler_from_quaternion
import tf2_ros

from rclpy.node import Node
import rclpy
import numpy as np

assert rclpy


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('num_particles', 200)

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)

        #  *Important Note #2:* You must respond to pose
        #   initialization requests sent to the /initialpose
        #   topic. You can test that this works properly using the
        #   "Pose Estimate" feature in RViz, which publishes to
        #    /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)
        self.particle_pub = self.create_publisher(PoseArray, "/particles", 1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Particle state
        self.particles = None
        self.weights = None

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def pose_callback(self, msg):
        """Initialize particles around a clicked pose in RViz."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        o = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion((o.x, o.y, o.z, o.w))

        self.particles = np.zeros((self.num_particles, 3))
        self.particles[:, 0] = x + np.random.normal(0, 0.5, self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, 0.5, self.num_particles)
        self.particles[:, 2] = theta + np.random.normal(0, 0.3, self.num_particles)

        self.weights = np.ones(self.num_particles) / self.num_particles
        self.motion_model.latest_odom = None

        self.get_logger().info("Particles initialized around (%.2f, %.2f, %.2f)" % (x, y, theta))

    def laser_callback(self, msg):
        """Use lidar scan to weight and resample particles."""
        if self.particles is None:
            return

        # Downsample the scan to match num_beams_per_particle
        ranges = np.array(msg.ranges)
        num_beams = self.sensor_model.num_beams_per_particle
        indices = np.linspace(0, len(ranges) - 1, num_beams, dtype=int)
        downsampled = ranges[indices]

        self.weights = self.sensor_model.evaluate(self.particles, downsampled)

        if self.weights is None:
            return

        # Squash weights to keep particle diversity
        self.weights = self.weights ** (1.0 / 3.0)

        # Normalize weights
        weight_sum = np.sum(self.weights)


        if weight_sum == 0:
            self.weights = np.ones(self.num_particles) / self.num_particles
        else:
            self.weights = self.weights / weight_sum

        # Resample particles based on weights
        indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            replace=True,
            p=self.weights
        )
        self.particles = self.particles[indices]

        # Add small noise after resampling to keep diversity
        self.particles[:, 0] += np.random.normal(0, 0.01, self.num_particles)
        self.particles[:, 1] += np.random.normal(0, 0.01, self.num_particles)
        self.particles[:, 2] += np.random.normal(0, 0.005, self.num_particles)

        # Publish estimated pose
        self.publish_pose()

    def publish_pose(self):
        """Compute average particle pose and publish."""
        if self.particles is None:
            return

        # To handle multi-modal distributions (where particles split into two distinct groups),
        # a simple average would pick an empty spot in the middle. Instead, we can find the
        # most dense cluster of particles using a 2D histogram, and only average that cluster.
        hist, xedges, yedges = np.histogram2d(self.particles[:, 0], self.particles[:, 1], bins=20)
        max_idx = np.unravel_index(np.argmax(hist), hist.shape)

        x_min, x_max = xedges[max_idx[0]], xedges[max_idx[0]+1]
        y_min, y_max = yedges[max_idx[1]], yedges[max_idx[1]+1]

        # Filter down to particles that are inside this most-dense bin
        in_cluster = (
            (self.particles[:, 0] >= x_min) & (self.particles[:, 0] <= x_max) &
            (self.particles[:, 1] >= y_min) & (self.particles[:, 1] <= y_max)
        )
        cluster = self.particles[in_cluster]

        # Fallback just in case floating point weirdness results in an empty cluster
        if len(cluster) == 0:
            cluster = self.particles

        avg_x = float(np.mean(cluster[:, 0]))
        avg_y = float(np.mean(cluster[:, 1]))
        avg_theta = float(np.arctan2(
            np.mean(np.sin(cluster[:, 2])),
            np.mean(np.cos(cluster[:, 2]))
        ))

        # Publish as Odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "/map"
        odom_msg.child_frame_id = self.particle_filter_frame

        odom_msg.pose.pose.position.x = avg_x
        odom_msg.pose.pose.position.y = avg_y
        odom_msg.pose.pose.position.z = 0.0

        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = float(np.sin(avg_theta / 2.0))
        odom_msg.pose.pose.orientation.w = float(np.cos(avg_theta / 2.0))

        self.odom_pub.publish(odom_msg)

        # Publish TF: map -> particle_filter_frame
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = "/map"
        t.child_frame_id = self.particle_filter_frame

        t.transform.translation.x = avg_x
        t.transform.translation.y = avg_y
        t.transform.translation.z = 0.0

        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = float(np.sin(avg_theta / 2.0))
        t.transform.rotation.w = float(np.cos(avg_theta / 2.0))

        self.tf_broadcaster.sendTransform(t)

        self.publish_particles()

    def publish_particles(self):
        """Publish particles as a PoseArray for RViz visualization."""
        if self.particles is None:
            return

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "/map"

        poses = []
        for x, y, theta in self.particles:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(np.sin(theta / 2.0))
            pose.orientation.w = float(np.cos(theta / 2.0))
            poses.append(pose)

        pose_array.poses = poses
        self.particle_pub.publish(pose_array)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()


# class ParticleFilter(Node):

#     def __init__(self):
#         super().__init__("particle_filter")

#         self.declare_parameter('particle_filter_frame', "default")
#         self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

#         #  *Important Note #1:* It is critical for your particle
#         #     filter to obtain the following topic names from the
#         #     parameters for the autograder to work correctly. Note
#         #     that while the Odometry message contains both a pose and
#         #     a twist component, you will only be provided with the
#         #     twist component, so you should rely only on that
#         #     information, and *not* use the pose component.

#         self.declare_parameter('odom_topic', "/odom")
#         self.declare_parameter('scan_topic', "/scan")

#         scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
#         odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

#         self.laser_sub = self.create_subscription(LaserScan, scan_topic, self.laser_callback, 1)

#         self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odom_callback, 1)

#         #  *Important Note #2:* You must respond to pose
#         #   initialization requests sent to the /initialpose
#         #   topic. You can test that this works properly using the
#         #   "Pose Estimate" feature in RViz, which publishes to
#         #    /initialpose.

#         self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
#                                                  self.pose_callback,
#                                                  1)

#         #  *Important Note #3:* You must publish your pose estimate to
#         #     the following topic. In particular, you must use the
#         #     pose field of the Odometry message. You do not need to
#         #     provide the twist part of the Odometry message. The
#         #     odometry you publish here should be with respect to the
#         #     "/map" frame.

#         self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

#         # Initialize the models
#         self.motion_model = MotionModel(self)
#         self.sensor_model = SensorModel(self)

#         self.get_logger().info("=============+READY+=============")

#         # Implement the MCL algorithm
#         # using the sensor model and the motion model
#         #
#         # Make sure you include some way to initialize
#         # your particles, ideally with some sort
#         # of interactive interface in rviz
#         #
#         # Publish a transformation frame between the map
#         # and the particle_filter_frame.

# def main(args=None):
#     rclpy.init(args=args)
#     pf = ParticleFilter()
#     rclpy.spin(pf)
#     rclpy.shutdown()
