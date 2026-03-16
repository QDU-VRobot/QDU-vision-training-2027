from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # 获取参数文件的完整路径,将config和yaml文件联系在一起
    params_file = os.path.join(
        get_package_share_directory('my_camera_cpp'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='my_camera_cpp',
            executable='image_publisher',
            name='image_publisher',
            output='screen'
        ),
        Node(
            package='my_camera_cpp',
            executable='image_subscriber',
            name='image_subscriber',
            parameters=[params_file],  # 加载YAML参数
            output='screen'
        )
    ])