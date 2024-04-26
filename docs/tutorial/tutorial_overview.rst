Tutorials and Use
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

.. tip::
    Want to use only a part of the data? You can pass e.g. ``data_selection = "TT only"`` when initialising the likelihood. See :ref:`Data Selection` for more information.

If you have a dictionary of parameter values and CMB spectra you can then go ahead and calculate the :math:`\chi^2`:

.. code-block:: python

    params = {"Dl": {"TT": .., "EE": .., "TE": ..}, ...}# followed by nuisance parameters
    chi2 = candl_like.chi_square(params)

.. note::

    The likelihood operates in :math:`D_\ell` space, i.e. on :math:`\ell (\ell + 1) C_\ell / (2 \pi)`, in units of :math:`\mu K_{\mathrm{CMB}}^2`.
    Theory spectra start at :math:`\ell=2`.

``traditional_tutorial.ipynb``
------------------------------

This notebook shows how traditional inference tasks are accomplished. In particular:

* Initialising the likelihood and accessing the data (band powers, covariance, etc.)
* Interfacing the likelihood with CAMB and calculating the :math:`\chi^2` for a given spectrum
* Interfacing the likelihood with Cobaya and running an MCMC chain

This tutorial uses some optional packages.
Make sure you have Cobaya, getdist, and CAMB installed in order to run the whole notebook.

``differentiable_tutorial.ipynb``
---------------------------------

This notebook shows different aspects relying on the differentiability of the likelihood. In particular:

* Initialising the likelihood and accessing the data (band powers, covariance, etc.)
* Running gradient-based minimisers
* Interfacing the likelihood with Optax
* Running NUTS chains by interfacing the likelihood with BlackJAX

This tutorial uses some optional packages.
Make sure you have Optax, BlackJAX, getdist, and CosmoPower-JAX installed in order to run the whole notebook.
You also need to have some emulator models for CosmoPower-JAX; we recommend the SPT high-accuracy models available `here <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/SPT_high_accuracy>`_.

Working With Instantiated Likelihoods
----------------------------------------------------------

In general, modifications to instantiated likelihood objects are only correctly propagated, if they are done immediately after initialisation.
This has to do with how JAX's jit works with class methods (more details can be found here `here <https://jax.readthedocs.io/en/latest/faq.html#how-to-use-jit-with-methods>`_).
While this may change in the future, exercise caution for now.

.. warning::

   Changes to the attributes of candl likelihoods (e.g. the band powers or the data model) must be made immediately after initialisation.
   Once a jitted method has been called (e.g. the likelihood has been evaluated), changes to attributes are no longer tracked.
   Therefore, it is advised to perform any customisation immediately after initialisation (or by modifying the underlying .yaml file directly - see :ref:`here<Data Structure>` for more info on how to do that).

Combining Multiple Likelihoods
----------------------------------------------------------

You can combine multiple likelihoods by defining a function that returns the sum of the individual likelihoods.
However, for this approach the data sets in question need to be independent of one another.
Consult the literature and in particular the relevant release papers to verify that this is true.
If this is not the case, you need to create a new data set, with a long data vector comprising of the individual data sets and account for the correlation between the data sets in the covariance matrix.
See :ref:`Data Structure` for more details on the structure of data sets.
Even if your data sets are independent, be sure to check that you are not applying any priors twice (e.g. on :math:`\tau`).
