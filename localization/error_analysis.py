#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class ErrorAnalysisNode(Node):
    def __init__(self):
        super().__init__('error_analysis')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.declare_parameter('gt_frame', 'base_link')
        self.declare_parameter('pf_frame', 'base_link_pf')
        self.declare_parameter('map_frame', 'map')
        
        self.gt_frame = self.get_parameter('gt_frame').value
        self.pf_frame = self.get_parameter('pf_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        
        self.error_history = []
        self.t_history = []
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.start_time = None
        
        self.get_logger().info("ErrorAnalysisNode Started. Ensure /map to /base_link and /base_link_pf transforms are active.")
        self.get_logger().info("When done, press Ctrl+C to generate the chart.")

    def timer_callback(self):
        try:
            # Look up latest transforms (lookup_transform with time=0 gets the newest)
            gt_trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.gt_frame, 
                rclpy.time.Time()
            )
            
            pf_trans = self.tf_buffer.lookup_transform(
                self.map_frame, 
                self.pf_frame, 
                rclpy.time.Time()
            )
            
            # Calculate Euclidean distance on XY plane
            dx = pf_trans.transform.translation.x - gt_trans.transform.translation.x
            dy = pf_trans.transform.translation.y - gt_trans.transform.translation.y
            error = np.sqrt(dx**2 + dy**2)
            
            now = self.get_clock().now().nanoseconds / 1e9
            if self.start_time is None:
                self.start_time = now
            t = now - self.start_time
            
            self.error_history.append(error)
            self.t_history.append(t)
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pass

    def plot_errors(self):
        if len(self.error_history) == 0:
            self.get_logger().info("No error data was collected. Transforms might not have been published.")
            return
            
        matplotlib.use('Agg') # Ensure plot generation doesn't block headless systems
        plt.figure(figsize=(10, 5))
        plt.plot(self.t_history, self.error_history, label='L2 Error (m)')
        
        avg_error = np.mean(self.error_history)
        plt.axhline(avg_error, color='red', linestyle='dashed', label=f'Avg Error: {avg_error:.3f}m')
        
        plt.title('Particle Filter Extrapolation Error vs Ground Truth')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance Error (m)')
        plt.grid(True)
        plt.legend()
        plt.savefig('pf_error_plot.png')
        self.get_logger().info("Saved final plot to 'pf_error_plot.png'")

def main(args=None):
    rclpy.init(args=args)
    node = ErrorAnalysisNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Caught Keyboard Interrupt. Writing output...")
    finally:
        node.plot_errors()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
