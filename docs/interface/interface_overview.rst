Interface
=================================================

.. toctree::
   :hidden:
   :titlesonly:
   :maxdepth: 1

   interface_theory
   interface_sampler

At its heart, the likelihood is a straightforward function that takes a set of parameters and CMB spectra and returns the log likelihood value.
Therefore, the interface is rather simple and allows candl to be used with a variety of other software.

The most common use case is to interface the likelihood with a :ref:`theory code<Theory Codes>`, i.e. a code that provides CMB spectra given a set of cosmological parameters, in order to create a function that moves from parameters to the log likelihood in one step.
The next step, is then to explore this function using a :ref:`sampler<Samplers and Minimisers>`, which explores the parameter space and finds the best-fit point or returns samples from the posterior distribution.
The respective pages contain more information on how both of these tasks are achieved.
A set of ready-made solutions for different popular software packages is provided in the ``interface`` module.
