Installation
============

You can install empymod either via ``conda``:

.. code-block:: console

   conda install -c conda-forge empymod

or via ``pip``:

.. code-block:: console

   pip install empymod

Requirements are the modules ``numpy>=2.0.0``, ``scipy``, ``numba``,
``libdlf``, and ``scooby``.

The modeller empymod comes with add-ons (``empymod.scripts``). These add-ons
provide some very specific, additional functionalities. Some of these add-ons
have additional, optional dependencies such as matplotlib. See the
*Add-ons*-section for their documentation. For interactive plots you will need
ipympl in addition to matplotlib.

If you are new to Python we recommend using a Python distribution, which will
ensure that all dependencies are met, specifically properly compiled versions
of ``NumPy`` and ``SciPy``; we recommend using `Anaconda
<https://www.anaconda.com/distribution>`_. If you install Anaconda you can
simply start the *Anaconda Navigator*, add the channel ``conda-forge`` and
``empymod`` will appear in the package list and can be installed with a click.

Using NumPy and SciPy with the Intel Math Kernel Library (*mkl*) can
significantly improve computation time. You can check if ``mkl`` is used via
``conda list``: The entries for the BLAS and LAPACK libraries should contain
something with ``mkl``, not with ``openblas``. To enforce it you might have to
create a file ``pinned``, containing the line ``libblas[build=*mkl]`` in the
folder ``path-to-your-conda-env/conda-meta/``.

.. note::

    Until v2 empymod did not use Numba but instead optionally NumExpr. Use
    **v1.10.x** if you cannot use Numba or want to use NumExpr. However, no
    new features will land in v1, only bugfixes.

.. admonition:: Julia wrapper

    A Julia wrapper for empymod was created by @ruboerner and is available from
    `github.com/ruboerner/EmpymodWrapper.jl <https://github.com/ruboerner/EmpymodWrapper.jl>`_.
