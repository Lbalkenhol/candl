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
We supply tools for a series of theory codes to be used in this way.

.. tip::
   For a differentiable pipeline from parameters to the log likelihood value, use CosmoPower-JAX.

CAMB
-------------------------------------------------

:Github: `CAMB <https://github.com/cmbant/CAMB/tree/master>`_

:Description: Classic Boltzmann solver

:Paper: |camb|

.. autofunction:: candl.interface.get_CAMB_pars_to_theory_specs_func

CLASS
-------------------------------------------------

:Github: `CLASS <https://github.com/lesgourg/class_public>`_

:Description: Classic Boltzmann solver

:Paper: |class|

.. autofunction:: candl.interface.get_CLASS_pars_to_theory_specs_func

CosmoPower
-------------------------------------------------

:Github: `CosmoPower <https://github.com/alessiospuriomancini/cosmopower>`_

:Description: Neural-network based emulator, TensorFlow implementation

:Paper: |cosmopower|

.. note::
   You can find trained CosmoPower models at:

   * Original :math:`\Lambda\mathrm{CDM}` models released with the CosmoPower paper (`here <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/CP_paper>`_)
   * :math:`\Lambda\mathrm{CDM}`, :math:`N_\mathrm{eff}`, :math:`A_L` models trained on high-accuracy CAMB spectra used in the `SPT-3G 2018 TT/TE/EE analysis <https://arxiv.org/abs/2212.05642>`_ (`here <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/CP_paper>`_)
   * :math:`\Lambda\mathrm{CDM}`, :math:`N_\mathrm{eff}`, :math:`\Sigma m_\nu`, :math:`\mathrm{wCDM}` models trained on high-accuracy CLASS spectra from `Bolliet et al. 2023 <https://arxiv.org/abs/2303.01591>`_ (`here <https://github.com/cosmopower-organization>`_)

.. autofunction:: candl.interface.get_CosmoPower_pars_to_theory_specs_func

CosmoPower-JAX
-------------------------------------------------

:Github: `CosmoPower-JAX <https://github.com/dpiras/cosmopower-jax>`_

:Description: Neural-network based emulator, JAX implementation

:Paper: |cosmopower_jax|

.. tip::
   Vanilla CosmoPower models can be loaded into CosmoPower-JAX.

.. autofunction:: candl.interface.get_CosmoPowerJAX_pars_to_theory_specs_func

Capse.jl
-------------------------------------------------

:Github: `Capse.jl <https://github.com/CosmologicalEmulators/Capse.jl>`_

:Description: Neural-network based emulator, Julia implementation with `pycapse wrapper <https://github.com/CosmologicalEmulators/pycapse>`_.

:Paper: |capse|

.. autofunction:: candl.interface.get_PyCapse_pars_to_theory_specs_func

Cobaya Theory Classes
-------------------------------------------------

Any ``cobaya.theory.Theory`` class (see the `Cobaya documentation <https://cobaya.readthedocs.io/en/latest/index.html>`_ for details) can be used.

.. autofunction:: candl.interface.get_CobayaTheory_pars_to_theory_specs_func
