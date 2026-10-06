from setuptools import find_packages, setup

package_name = 'pinky_multi_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'domain52_central_bridge.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tory',
    maintainer_email='tory@example.com',
    description='Nav2 goal coordinator and mission action server for two Pinky robots.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'two_pinky_coordinator = pinky_multi_robot.two_pinky_coordinator:main',
            'mission_action_server = pinky_multi_robot.mission_action_server:main',
            'nav2_action_proxy = pinky_multi_robot.nav2_action_proxy:main',
            'nav2_map_ui = pinky_multi_robot.nav2_map_ui:main',
        ],
    },
)
