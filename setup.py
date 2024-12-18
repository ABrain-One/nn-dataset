from setuptools import setup, find_packages
from pathlib import Path

# Function to read the requirements.txt file
def read_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Safely read the README.md file
def read_readme():
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return ""

setup(
    name="nn-dataset",
    version="0.1.0",
    description="A neural network dataset management tool",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="ABrain-One",
    author_email="dmytro.ignatov@uni-wuerzburg.de",
    url="https://github.com/ABrain-One/nn-dataset",
    packages=find_packages(include=["ab.*"]),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
