.. |camb| image:: https://img.shields.io/badge/arXiv-9911177-b31b1b.svg
   :target: https://arxiv.org/abs/astro-ph/9911177

.. |class| image:: https://img.shields.io/badge/arXiv-1104.2933-b31b1b.svg
   :target: https://arxiv.org/abs/1104.2933

.. |cosmopower| image:: https://img.shields.io/badge/arXiv-2106.03846-b31b1b.svg
   :target: https://arxiv.org/abs/2106.03846

.. |cosmopower_jax| image:: https://img.shields.io/badge/arXiv-2305.06347-b31b1b.svg
   :target: https://arxiv.org/abs/2305.06347

.. |capse| image:: https://img.shields.io/badge/arXiv-2307.14339-b31b1b.svg
   :target: https://arxiv.org/abs/2307.14339

Theory Codes
=================================================

Theory codes perform the calculation of the CMB spectrum given cosmological parameter values.
For the purpose of interfacing with them, we reduce them to a single function call, taking as an input a dictionary of parameter names and values and producing as an output a dictionary of CMB spectra.
We supply tools that allow for the following theory codes to be used in this way:

* `CAMB <https://github.com/cmbant/CAMB/tree/master>`_ |camb|
* `CLASS <https://github.com/lesgourg/class_public>`_ |class|
* `CosmoPower <https://github.com/alessiospuriomancini/cosmopower>`_ |cosmopower|
* `CosmoPower-JAX <https://github.com/dpiras/cosmopower-jax>`_ |cosmopower_jax|
* `Capse.jl <https://github.com/CosmologicalEmulators/Capse.jl>`_ (through the `pycapse <https://github.com/CosmologicalEmulators/pycapse>`_ wrapper) |capse|
* Any ``cobaya.theory.Theory`` class (see the `Cobaya documentation <https://cobaya.readthedocs.io/en/latest/index.html>`_ for details).

For differentiability from parameters to the log likelihood value, we advise the use CosmoPower-JAX, though other emulators may also work.
See the ``interface`` module of how each theory code is used and the tutorial for examples.
