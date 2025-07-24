from setuptools import setup, find_packages

setup(
    name='speechscore',
    version='0.1.0',
    packages=find_packages(),  # automatically finds subpackages
    install_requires=[],       # dependencies (e.g., ['numpy', 'scipy'])
    author='ClearerVoice-Studio',
    description='Compute speech quality metrics',
)