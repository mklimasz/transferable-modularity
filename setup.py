from setuptools import setup, find_packages

setup(
    name='modular-kd',
    version='0.0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)
