# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup

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
        "Programming Language :: Python",
    ],
    entry_points={
        "console_scripts": [
            "empymod=empymod.__main__:main",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "scipy>=1.9",
        "numpy<2.0",
        "numba>=0.53",
        "libdlf",
        "scooby",
    ],
    use_scm_version={
        "root": ".",
        "relative_to": __file__,
        "write_to": os.path.join("empymod", "version.py"),
    },
    setup_requires=["setuptools_scm"],
)
