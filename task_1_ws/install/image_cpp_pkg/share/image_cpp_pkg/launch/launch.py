import launch
import launch_ros
import os                   #处理文件路径
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    params_file = os.path.join(     #自动组合路径
        get_package_share_directory('image_cpp_pkg'),#获取包的共享目录
        'config',
        'image_params.yaml'
    ) 

    """产生launch描述"""
    action_ImagePublishNode_node=launch_ros.actions.Node(
        package='image_cpp_pkg',
        executable='ImagePublishNode',
        parameters=[params_file],   
        output='screen'
    )
    action_ImageSubscribeNode_node=launch_ros.actions.Node(
        package='image_cpp_pkg',
        executable='ImageSubscribeNode',
        output='screen'
    )
    return launch.LaunchDescription([
        #actions
        action_ImagePublishNode_node,
        action_ImageSubscribeNode_node
    ])
