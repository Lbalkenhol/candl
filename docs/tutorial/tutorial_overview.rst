Tutorials
=================================================

We provide two jupyter notebooks that show the likelihood in action.
You can find them in the ``notebooks/`` on the git repo.

``traditional_tutorial.ipynb``
------------------------------

This notebook shows how traditional inference tasks are accomplished. In particular:

* Initialising the likelihood and accessing the data (band powers, covariance, etc.)
* Interfacing the likelihood with CAMB and calculating the :math:`\chi^2` for a given spectrum
* Interfacing the likelihood with Cobaya and running an MCMC chain


``differentiable_tutorial.ipynb``
---------------------------------

This notebook shows different aspects relying on the differentiability of the likelihood. In particular:

* Initialising the likelihood and accessing the data (band powers, covariance, etc.)
* Running gradient-based minimisers
* Interfacing the likelihood with Optax
* Running NUTS chains by interfacing the likelihood with BlackJAX
