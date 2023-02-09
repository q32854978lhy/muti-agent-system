from setuptools import setup

setup(
    name='collisionavoidance',
    version='2020',
    packages=[
        'coll_avo',
        'coll_avo.parameters',
        'coll_avo.methods',
        'coll_avo.utils',
        'coll_gym',
        'coll_gym.envs',
        'coll_gym.envs.methods',
        'coll_gym.envs.utils',
    ],
    install_requires=[
        'gitpython',
        'gym',
        'matplotlib',
        'numpy',
        'scipy',
        'torch',
        'torchvision',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
    },
)
