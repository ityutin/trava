import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="trava",
    version="0.2.10",
    description="Framework that helps to train models, compare them and track parameters&metrics along the way.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/ityutin/trava",
    author="Ilya Tyutin",
    author_email="emmarrgghh@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(
        exclude=("*.tests", "*.tests.*", "tests.*", "tests", "*.examples", "*.examples.*", "examples.*", "examples")
    ),
    include_package_data=True,
    install_requires=["pandas", "numpy"],
)
