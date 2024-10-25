from setuptools import find_packages, setup

package_name = 'single_object'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'torch', 'matplotlib', 'opencv-python', 'sam2'],
    zip_safe=True,
    maintainer='fyp',
    maintainer_email='howard.tse2005@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'single_object_node = single_object.single_object:main'
        ],
    },
)
