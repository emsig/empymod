# -*- coding: utf-8 -*-
import os
import re
import sys
from setuptools import setup

if not sys.version_info[:2] >= (3, 8):
    sys.exit(f"empymod is only meant for Python 3.8 and up.\n"
             f"Current version: {sys.version_info[0]}.{sys.version_info[1]}.")

# Get README and remove badges.
with open("README.rst") as f:
    readme = re.sub(r"\|.*\|", "", f.read(), flags=re.DOTALL)

description = "Open-source full 3D electromagnetic modeller for 1D VTI media"

setup(
    name="empymod",
    description=description,
    long_description=readme,
    author="The emsig community",
    author_email="info@emsig.xyz",
    url="https://emsig.xyz",
    license="Apache-2.0",
    packages=["empymod", "empymod.scripts"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        "console_scripts": [
            "empymod=empymod.__main__:main",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "scipy>=1.5",
        "numba>=0.50",
    ],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": os.path.join("empymod", "version.py"),
    },
    setup_requires=["setuptools_scm"],
)
