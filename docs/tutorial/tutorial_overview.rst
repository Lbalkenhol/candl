Tutorials
=================================================

Below, you can find an example of how to quickly get started with candl.
In addition, we provide two jupyter notebooks that show the likelihood in action.
You can find them in the ``notebooks/`` on the git repo.

Quickstart
------------------------------

To initialise the likelihood, point it to the data set ``yaml`` file.
Short cuts exist for released data sets.
For example, let's say we want to work with the SPT-3G 2018 TT/TE/EE data set.

.. code-block:: python

    import candl
    import candl.data
    
    candl_like = candl.Like(candl.data.SPT3G_2018_TTTEEE)

and that's it! You can now access aspects of the data, for example the band powers ``candl_like.data_bandpowers`` and the covariance matrix ``candl_like.covariance``.
If you have a dictionary of parameter values and CMB spectra you can then go ahead and calculate the :math:`\chi^2`:

.. code-block:: python

    params = {"Dl": {"TT": .., "EE": .., "TE": ..}, ...}# followed by nuisance parameters
    chi2 = candl_like.chi_square(params)

.. note::

    The likelihood operates in :math:`D_\ell` space, i.e. on :math:`C_\ell \ell (\ell + 1) / (2 \pi)`, in units of :math:`\mu K_{\mathrm{CMB}}^2`.
    Theory spectra start at :math:`\ell=2`.

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
