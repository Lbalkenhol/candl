.. image:: https://github.com/Lbalkenhol/candl/raw/main/docs/logos/candl_wordmark&symbol_col_RGB.png
    :width: 800

.. |docsshield| image:: https://img.shields.io/readthedocs/candl
   :target: http://candl.readthedocs.io

.. |arxivshield| image:: https://img.shields.io/badge/arXiv-2401.13433-b31b1b.svg
   :target: https://arxiv.org/abs/2401.13433

CMB Analysis With A Differentiable Likelihood
===============================================================

:Authors: L\. Balkenhol, C\. Trendafilova, K\. Benabed, S\. Galli

:Paper: |arxivshield|

:Source: `<https://github.com/Lbalkenhol/candl>`__

:Documentation: |docsshield|

candl is a differentiable likelihood framework for analysing CMB power spectrum measurements.
Key features are:

* JAX-compatibility, allowing for fast and easy computation of gradients and Hessians of the likelihoods.
* The latest public data releases from the South Pole Telescope and Atacama Cosmology Telescope collaborations.
* Interface tools for work with other popular cosmology software packages (e.g. Cobaya and MontePython).
* Auxiliary tools for common analysis tasks (e.g. generation of mock data).

candl supports the analysis of primary CMB and lensing power spectrum data (:math:`TT`, :math:`TE`, :math:`EE`, :math:`BB`, :math:`\phi\phi`, :math:`\kappa\kappa`).

Installation
------------

candl can be installed with pip::

    pip install candl-like

After installation, we recommend testing by executing the following python code::

    import candl.tests
    candl.tests.run_all_tests()

This well test all data sets included in candl.

Data Sets
------------

candl data sets are kept in the `candl_data <https://github.com/lbalkenhol/candl_data>`__ repository. Detailed installation instructions for the data can be found on the dedicated repo page, but in short, you navigate to where you would like to store the data and then run::

    git clone https://github.com/Lbalkenhol/candl_data.git
    cd candl_data
    pip install .

The following data sets are available:

* SPT-3G 2018 TT/TE/EE (`Balkenhol et al. 2023 <https://arxiv.org/abs/2212.05642>`__)
* SPT-3G 2018 Lensing (`Pan et al. 2023 <https://arxiv.org/abs/2308.11608>`__)
* ACT DR6 TT/TE/EE (`Naess et al. 2025 <https://arxiv.org/abs/2503.14451>`__, `Louis et al. 2025 <https://arxiv.org/abs/2503.14452>`__, `Calabrese et al. 2025 <https://arxiv.org/abs/2503.14454>`__)
* ACT DR4 TT/TE/EE (`Aiola et al. 2020 <https://arxiv.org/abs/2007.07288>`__, `Choi et al. 2020 <https://arxiv.org/abs/2007.07289>`__)
* ACT DR6 Lensing (`Madhavacheril et al. 2023 <https://arxiv.org/abs/2304.05203>`__, `Qu et al. 2023 <https://arxiv.org/abs/2304.05202>`__)
* Planck likelihoods available through `clipy <https://github.com/benabed/clipy>`__

Detailed information on these data sets and instructions on how you can add your own can be found `in the docs <https://candl.readthedocs.io/en/latest/data/data_overview.html>`__.

JAX
---

`JAX <https://github.com/google/jax>`__ is a Google-developed python library.
In its own words: *"JAX is Autograd and XLA, brought together for high-performance numerical computing."*

candl is written in a JAX-friendly way.
That means JAX is optional and you can install and run candl without JAX and perform traditional inference tasks such as MCMC sampling with Cobaya.
However, if JAX is installed, the likelihood is fully differentiable thanks to automatic differentiation and many functions are jitted for speed.

Packages and Versions
---------------------------

candl has been built on python 3.10.
You may be able to get it running on 3.9, but this is not officially supported - run it at your own risk.

candl has been tested on JAX versions 0.4.31 and 0.4.24.

Documentation
--------------

You can find the documentation `here <http://candl.readthedocs.io>`_.

Citing candl
--------------

If you use candl please cite the `release paper <https://arxiv.org/abs/2401.13433>`_. Be sure to also cite the relevant papers for any samplers, theory codes, and data sets you use.

===================

.. |cnrs| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/cnrs_logo.jpeg
   :alt: CNRS
   :height: 100px
   :width: 100px

.. |erc| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/erc_logo.jpeg
   :alt: ERC
   :height: 100px
   :width: 100px

.. |NEUCosmoS| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/neucosmos_logo.png
   :alt: NEUCosmoS
   :height: 100px
   :width: 159px

.. |IAP| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/IAP_logo.jpeg
   :alt: IAP
   :height: 100px
   :width: 104px

.. |Sorbonne| image:: https://github.com/Lbalkenhol/candl/raw/main/logos/sorbonne_logo.jpeg
   :alt: Sorbonne
   :height: 100px
   :width: 248px

|cnrs| |erc| |NEUCosmoS| |IAP| |Sorbonne|
