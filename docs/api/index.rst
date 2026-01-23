.. _api:

#############
API reference
#############

:Release: |version|
:Date: |today|

----

.. module:: empymod

.. toctree::
   :maxdepth: 2
   :hidden:

   filters
   kernel
   model
   transform
   utils
   io
   fdesign
   tmtemod


.. grid:: 1
    :gutter: 2

    .. grid-item-card::

        Arbitrary oriented, finite-length dipoles: :func:`empymod.model.bipole`

    .. grid-item-card::

        Infinitesimal small, grid-oriented dipoles:
        :func:`empymod.model.dipole`

    .. grid-item-card::

        Analytical full- and half-space solutions:
        :func:`empymod.model.analytical`

    .. grid-item-card::

        In-phase and Quadrature: :func:`empymod.model.ip_and_q`
