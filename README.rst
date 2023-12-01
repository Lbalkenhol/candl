.. image:: docs/logos/candl_wordmark&symbol_col_RGB.png
    :width: 800

CMB Analysis With A Differentiable Likelihood
===============================================================

:Authors: ``us!``

:Paper: ``arxiv shield goes here``

:Source: `<https://github.com/Lbalkenhol/candl>`__

:Documentation: ``right here``

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

    pip install candl_like

JAX
---

`JAX <https://github.com/google/jax>`__ is a Google-developed python library.
In its own words: *"JAX is Autograd and XLA, brought together for high-performance numerical computing."*

candl is written in a JAX-friendly way.
That means JAX is optional and you can install and run candl without JAX and perform traditional inference tasks such as MCMC sampling with Cobaya.
However, if JAX is installed, the likelihood is fully differentiable thanks to automatic differentiation and many functions are jitted for speed.

===================

.. image:: https://github.com/Lbalkenhol/candl/blob/7519bd69b29395f18e6721c3a940c9d1ec898f8a/docs/logos/cnrs_logo.jpeg
   :alt: CNRS
   :height: 100px

.. image:: https://github.com/Lbalkenhol/candl/blob/7519bd69b29395f18e6721c3a940c9d1ec898f8a/docs/logos/erc_logo.jpeg
   :alt: ERC
   :height: 100px

.. image:: https://raw.githubusercontent.com/Lbalkenhol/candl/main/docs/logos/neucosmos_logo.png
   :alt: NEUCosmoS
   :height: 100px

.. image:: https://github.com/Lbalkenhol/candl/blob/7519bd69b29395f18e6721c3a940c9d1ec898f8a/docs/logos/IAP_logo.jpeg
   :alt: IAP
   :height: 100px

.. image:: https://github.com/Lbalkenhol/candl/blob/7519bd69b29395f18e6721c3a940c9d1ec898f8a/docs/logos/sorbonne_logo.jpeg
   :alt: Sorbonne
   :height: 100px
