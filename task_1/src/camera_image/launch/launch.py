import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 获取参数文件的完整路径
    config = os.path.join(
        get_package_share_directory('camera_image'),
        'config',
        'params.yaml'
    )

    # 图像发布节点
    publisher_node = Node(
        package='camera_image',
        executable='image_publish',   # 注意：如果你的可执行文件名不同，请修改
        name='image_publisher',
        parameters=[config]
    )

    # 图像订阅节点
    subscriber_node = Node(
        package='camera_image',
        executable='image_subscribe',   # 你的订阅节点可执行文件名
        name='image_subscriber',
        parameters=[config]
    )

    return LaunchDescription([
        publisher_node,
        subscriber_node
    ])