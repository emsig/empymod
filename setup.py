# -*- coding: utf-8 -*-
from setuptools import setup
from empymod import __version__

try:
    import pypandoc
    readme = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    readme = open('README.md').read()

setup(
    name='empymod',
    version=__version__,
    description='Open-source full 3D electromagnetic modeller for 1D VTI media',
    long_description=readme,
    author='Dieter Werthmüller',
    author_email='dieter@werthmuller.org',
    url='https://empymod.github.io',
    download_url='https://github.com/empymod/empymod/tarball/v' + __version__,
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
