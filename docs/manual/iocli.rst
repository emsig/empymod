I/O & CLI
#########

Starting with ``empymod v2.2.0`` there are some basic saving and loading
routines together with a command line interface. This makes it possible to
model EM responses relatively straight forward from any other code.


.. _I/O:

I/O
---

There are two saving and two loading routines, one each for inputs and one for
data. «Input» in this context means all the parameters a modelling routine
takes. The saving/loading routines are on one hand good for persistence and
reproducibility, but on the other hand also necessary for the command-line
interface (see section `CLI`_).

``{save;load}_input``
~~~~~~~~~~~~~~~~~~~~~

Let's look at a simple example. From the start, we collect the input parameters
in a dictionary instead of providing them directly to the function.

.. ipython::

  In [1]: import empymod
     ...: import numpy as np
     ...:
     ...: # Define input parameters
     ...: inp = {
     ...:     'src': [[0, 0], [0, 1000], 250, [0, 90], 0],
     ...:     'rec': [np.arange(1, 6)*2000, np.zeros(5), 300, 0, 0],
     ...:     'freqtime': np.logspace(-1, 1, 3),
     ...:     'depth': [0, 300, 1500, 1600],
     ...:     'res': [2e14, 0.3, 1, 100, 1],
     ...: }
     ...:
     ...: # Model it
     ...: efield = empymod.bipole(**inp)

We can now easily save this dictionary to disk:


.. ipython::

  In [1]: empymod.io.save_input('myrun.json', inp)

This will save the input parameters in the file ``myrun.json`` (you can provide
absolute or relative paths in addition to the file name). The file name must
currently include ``.json``, as it is the only backend implemented so far. The
json-file is a plain text file, so you can open it with your favourite editor.
Let's have a look at it:

.. ipython::

  In [1]: !cat myrun.json

As you can see, it is basically the dictionary written as json. You can
therefore write your input parameter file with any program you want to.

These input files can then be loaded to run the *exactly* same simulation
again.

.. ipython::

  In [1]: inp_loaded = empymod.io.load_input('myrun.json')
     ...: efield2 = empymod.bipole(**inp_loaded)
     ...: # Let's check if the result is indeed the same.
     ...: print(f"Result is identical: {np.allclose(efield, efield2, atol=0)}")


``{save;load}_data``
~~~~~~~~~~~~~~~~~~~~

These functions are to store or load data. Using the computation from above,
we can store the data in one of the following two ways, either as json or as
text file:

.. ipython::

  In [1]: empymod.io.save_data('mydata.json', efield)
     ...: empymod.io.save_data('mydata.txt', efield)


Let's have a look at the text file:

.. ipython::

  In [1]: !cat mydata.txt

First is a header with the date, the version of empymod with which it was
generated, and the shape and type of the data. The columns are the sources (two
in this case), and in the rows there are first all receivers for the first
frequency (or time), then all receivers for the second frequency (or time), and
so on.

The json file is very similar. Here we print just the first twenty lines as an
example:

.. ipython::

  In [1]: !head -n 20 mydata.json

The main difference, beside the structure, is that the json-format does not
support complex data. It lists therefore first all real parts, and then all
imaginary parts. If you load it with another json reader it will therefore
have the dimension ``(2, nfreqtime, nrec, nsrc)``, where the 2 stands for real
and imaginary parts. (Only for frequency-domain data of course, not for
time-domain data.)

To load it in Python simply use the corresponding functions:

.. ipython::

  In [1]: efield_json = empymod.io.load_data('mydata.json')
     ...: efield_txt = empymod.io.load_data('mydata.txt')
     ...: # Let's check they are the same as the original.
     ...: print(f"Json-data: {np.allclose(efield, efield_json, atol=0)}")
     ...: print(f"Txt-data : {np.allclose(efield, efield_txt, atol=0)}")


Caution
~~~~~~~

There is a limitation to the ``save_input``-functionality: The data *must* be
three dimensional, ``(nfreqtime, nrec, nsrc)``. Now, in the above example that
is the case, we have 3 frequencies, 5 receivers, and 2 sources. However, if any
of these three quantities would be 1, empymod would by default squeeze the
dimension. To avoid this, you have to pass the keyword ``squeeze=False`` to the
empymod-routine.


.. _CLI:

CLI
---

The command-line interface is implemented for the top-level modelling routines
:func:`empymod.model.bipole`, :func:`empymod.model.dipole`,
:func:`empymod.model.loop`, and :func:`empymod.model.analytical`. To call it
you must write a json-file containing all your input parameters as described in
the section `I/O`_. The basic syntax of the CLI is

.. code-block:: console

   empymod <routine> --input <file> --output <file>

You can find some description as well by running the regular help

.. code-block:: console

   empymod --help

As an example, to reproduce the example given above in the I/O-section, run

.. code-block:: console

   empymod bipole --input myrun.json --output mydata.txt

If you do not specify ``--output`` the result will be printed to the STDOUT.


Warning re runtime
~~~~~~~~~~~~~~~~~~

A warning with regards to runtime: The CLI has an overhead, as it has to load
Python and empymod with all its dependencies each time (which is cached if
running in Python). Currently, the overhead should be less than 1s, and it will
come down further with changes happening in the dependencies. For doing some
simple forward modelling that should not be significant. However, it would
potentially be a bad idea to use the CLI for a forward modelling kernel in an
inversion. The inversion would spend a significant if not most of its time
starting Python and importing empymod over and over again.

Consult the following issue if you are interested in the overhead and its
status: `github.com/emsig/empymod/issues/162
<https://github.com/emsig/empymod/issues/162>`_.
