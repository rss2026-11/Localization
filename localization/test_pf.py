import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
import time

def publish_fake_data():
    rclpy.init()
    node = Node('fake_data_pub')
    map_pub = node.create_publisher(OccupancyGrid, '/map', 10)
    pose_pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    scan_pub = node.create_publisher(LaserScan, '/scan', 10)
    
    # Wait for connections
    time.sleep(1)
    
    # map
    m = OccupancyGrid()
    m.info.resolution = 0.05
    m.info.width = 100
    m.info.height = 100
    m.data = [0] * 10000
    map_pub.publish(m)
    
    time.sleep(0.5)
    
    # pose
    p = PoseWithCovarianceStamped()
    p.pose.pose.orientation.w = 1.0
    pose_pub.publish(p)
    
    time.sleep(0.5)
    
    # scan
    s = LaserScan()
    s.range_max = 10.0
    s.ranges = [5.0] * 1080 
    
    for _ in range(3):
        scan_pub.publish(s)
        time.sleep(0.1)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    publish_fake_data()
