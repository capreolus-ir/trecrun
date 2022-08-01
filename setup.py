import os
import setuptools

from setuptools.command.develop import develop
from setuptools.command.install import install


with open("README.md", "r") as fh:
    long_description = fh.read()

# from https://packaging.python.org/guides/single-sourcing-package-version/
def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "rt") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name="trecrun",
    version=get_version("trecrun/__init__.py"),
    author="Andrew Yates",
    author_email="one-name-then-the-next@gmail.com",
    description="Library for working with TREC run files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/capreolus-ir/trecrun",
    packages=setuptools.find_packages(),
    install_requires=[req.strip() for req in open("requirements.txt", "rt")],
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    python_requires=">=3.6",
    include_package_data=True,
)
