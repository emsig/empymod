# -*- coding: utf-8 -*-
from setuptools import setup

readme = open('README.rst').read()

description = 'Open-source full 3D electromagnetic modeller for 1D VTI media'

setup(
    name='empymod',
    version='1.8.3',
    description=description,
    long_description=readme,
    author='Dieter Werthmüller',
    author_email='dieter@werthmuller.org',
    url='https://empymod.github.io',
    download_url='https://github.com/empymod/empymod/tarball/v1.8.3',
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
