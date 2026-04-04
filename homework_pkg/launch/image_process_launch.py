from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('save_image'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='save_image',
            executable='ImagePublishNode',
            name='image_publisher_node',
            parameters=[config]
        ),
        Node(
            package='save_image',
            executable='ImageSubscribeNode',
            name='image_subscriber_node'
        )
    ])