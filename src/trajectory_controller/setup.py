from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'trajectory_controller'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=('test',)),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    # keep setuptools as requirement
    install_requires=['setuptools'],
    # **Important**: use 'scripts' to force install of wrappers into install/bin
    scripts=[
        'src/trajectory_controller/scripts/path_smoother_node',
        'src/trajectory_controller/scripts/trajectory_generator_node',
        'src/trajectory_controller/scripts/trajectory_tracker_node',
        'src/trajectory_controller/scripts/visualizer_node',
    ],
    zip_safe=True,
    maintainer='Student',
    maintainer_email='student@example.com',
    description='Path smoothing and trajectory control package',
    license='MIT',
    tests_require=['pytest'],
    # keep entry_points too (optional; scripts will guarantee wrappers exist)
    entry_points={
        'console_scripts': [
            'path_smoother_node = trajectory_controller.path_smoother:main',
            'trajectory_generator_node = trajectory_controller.trajectory_generator:main',
            'trajectory_tracker_node = trajectory_controller.trajectory_tracker:main',
            'visualizer_node = trajectory_controller.visualizer:main',
        ],
    },
)