import os
from launch import LaunchDescription
from launch_ros.actions import Node;
from ament_index_python.packages import get_package_share_directory as ge
def generate_launch_description():
    conf=os.path.join(ge('seeker'),'config','con.yaml')
    return LaunchDescription([
        Node(
            package='seeker',
            executable='publisher',
            name='pub_node',
            parameters=[conf],
            output='screen',
            emulate_tty=True,
        ),
        Node(
            package='seeker',
            executable='subscriber',
            name='sub_node',
            parameters=[conf],
            output='screen',
            emulate_tty=True,
        )
    ])
    