from setuptools import find_packages, setup

package_name = 'multiple_objects'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'torch', 'matplotlib', 'opencv-python', 'segment-anything'],
    zip_safe=True,
    maintainer='fyp',
    maintainer_email='howard.tse2005@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'multiple_objects_node = multiple_objects.multiple_objects:main'
        ],
    },
)
