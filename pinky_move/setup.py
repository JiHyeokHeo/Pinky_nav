from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'pinky_move'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.json')),
        (os.path.join('share', package_name, 'docs'),
            glob('docs/*.html') + glob('docs/*.md')),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tory',
    maintainer_email='vndj1212@gmail.com',
    description='Guarded tabletop patrol controller for Pinky Pro',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ir_adc_sim_node = pinky_move.ir_adc_sim_node:main',
            'csi_camera_node = pinky_move.csi_camera_node:main',
            'lane_autonomy = pinky_move.lane_autonomy:main',
            'lane_inference_worker = pinky_move.lane_inference_worker:main',
            'move_node = pinky_move.move_node:main',
            'factory_patrol = pinky_move.factory_patrol:main',
        ],
    },
)
