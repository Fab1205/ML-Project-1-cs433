from setuptools import find_packages, setup

setup(
    name="ml_project1",  # Replace with your package's name
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
