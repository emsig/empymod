# -*- coding: utf-8 -*-
from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

setup(
    name='empymod',
    version='1.2.1',
    description='ElectroMagnetic Python forward MODeller (1D)',
    long_description=readme,
    author='Dieter Werthmüller',
    author_email='dieter@werthmuller.org',
    url='https://github.com/prisae/empymod',
    download_url='https://github.com/prisae/empymod/tarball/v1.2.1',
    license='Apache License V2.0',
    packages=['empymod'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['numpy', 'scipy'],
)
