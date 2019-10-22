# -*- coding: utf-8 -*-
import re
from setuptools import setup

# Get README and remove badges.
readme = open('README.rst').read()
readme = re.sub('----.*marker', '----', readme, flags=re.DOTALL)

description = 'Open-source full 3D electromagnetic modeller for 1D VTI media'

setup(
    name='empymod',
    version='1.10.1.dev0',
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
)
