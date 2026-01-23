from setuptools import setup

package_name = 'http_action_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'flask'],
    zip_safe=True,
    maintainer='Newton',
    maintainer_email='newton@ssi.com',
    description='HTTP-to-ROS2 bridge for voice action commands',
    license='MIT',
    entry_points={
        'console_scripts': [
            'bridge_node = http_action_bridge.bridge_node:main',
        ],
    },
)
