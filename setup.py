# -*- coding: utf-8 -*-
from setuptools import setup

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='empymod',
    version='1.2.dev',
    description='ElectroMagnetic Python forward MODeller (1D)',
    long_description=readme,
    author='Dieter Werthmüller',
    author_email='dieter@werthmuller.org',
    url='https://github.com/prisae/empymod',
    license=license,
    packages=['empymod']
)
