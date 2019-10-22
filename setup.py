# -*- coding: utf-8 -*-
import os
import re
from setuptools import setup

# Get README and remove badges.
readme = open('README.rst').read()
readme = re.sub('----.*marker', '----', readme, flags=re.DOTALL)

description = 'Open-source full 3D electromagnetic modeller for 1D VTI media'

setup(
    name='empymod',
    description=description,
    long_description=readme,
    author='The empymod Developers',
    author_email='dieter@werthmuller.org',
    url='https://empymod.github.io',
    license='Apache License V2.0',
    packages=['empymod', 'empymod.scripts'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[
        'numpy',
        'scipy>=1.0.0',
    ],
    use_scm_version={
        'root': '.',
        'relative_to': __file__,
        'write_to': os.path.join('empymod', 'version.py'),
    },
    setup_requires=['setuptools_scm'],
)
