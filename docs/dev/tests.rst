Tests and Benchmarks
====================

If you ended up here you probably think about fixing some bugs or contributing
some code. **Awesome!** Just open a PR, and we will guide you through the
process. The following section contains some more detailed information of the
continues integration (CI) procedure we follow. In the end, each commit has to
pass them before it can be merged into the main branch on GitHub.


The first step to develop code is to clone the GitHub repo locally:

.. code-block:: console

   git clone git@github.com:emsig/empymod.git

All requirements for the dev-toolchain are collected in the
``requirements-dev.txt`` file, so you can install them all by running

.. code-block:: console

   pip install -r requirements_dev.txt

With this you have all the basic tools to run the tests, lint your code, build
the documentation, and so on.

Continuous Integration
----------------------

The CI elements are:

1. Linting: ``flake8``
2. Tests: ``pytest``
3. Code coverage: ``coveralls``
4. Link checks: ``sphinx``
5. Code quality: ``codacy``
6. Documentation: ``sphinx``
7. Benchmarks: ``asv``


(1) to (6) are run automatically through GitHub actions when committing changes
to GitHub. Any code change should pass these tests. Additionally, it is crucial
that new code comes with the appropriate tests and documentation, and if
applicable also with the appropriate benchmarks. However, you do not need any
of that to start a PR - everything can go step-by-step!

Many of the tests are set up in the Makefile (only tested on Linux):

- To install the current branch in editable mode:

  .. code-block:: console

     make install

- To check linting:

  .. code-block:: console

     make flake8

- To run pytest:

  .. code-block:: console

     make pytest

- To build the documentation:

  .. code-block:: console

     make html

- Or to list all the possibilities, simply run:

  .. code-block:: console

     make

There is also a benchmark suite using *airspeed velocity*, located in the
`emsig/empymod-asv <https://github.com/emsig/empymod-asv>`_-repository. The
results of my machine can be found in the `emsig/empymod-bench
<https://github.com/emsig/empymod-bench>`_, its rendered version at
`emsig.xyz/empymod-asv <https://emsig.xyz/empymod-asv>`_. They ensure that we
do not slow than the computation by introducing regressions.

