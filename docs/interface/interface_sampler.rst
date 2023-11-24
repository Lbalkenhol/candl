Samplers and Minimisers
=================================================

We provide tools to interface the likelihood with the two most commonly used MCMC samplers in cosmology: `Cobaya <https://github.com/CobayaSampler/cobaya>`_ and `MontePython <https://baudren.github.io/montepython.html>`_.
Moreover, we show how the likelihood can be interfaced with tools that explicitly exploit its differentiability: `BlackJAX <https://github.com/blackjax-devs/blackjax>`_ and `Optax <https://github.com/google-deepmind/optax>`_.

*Note:* be careful to only apply priors once.
For many data sets candl includes Gaussian priors on cosmological and nuisance parameters, to ensure a fully differentiable pipeline.
Some MCMC samplers offer to apply priors themselves, which can lead to double-counting.

Cobaya
-------------------------------------------------

To interface with Cobaya interactively, you can use the supplied interface code.
For example:

.. code-block:: python

    import candl
    import candl.data
    import candl.interface
    
    candl_like = candl.Like(candl.data.SPT3G_2018_TTTEEE)
    cobaya_dict = {"likelihood": candl.interface.get_cobaya_info_dict_for_like(candl_like)}

This will interface the likelihood with Cobaya and register all of its requirements.
You can then proceed to populate ``cobaya_dict`` with the the parameters to be sampled etc. and run Cobaya as usual.

In order to run Cobaya from the command line it sufficies to include the following lines in the likelihood block.

.. code-block:: yaml

    likelihood:
        candl.interface.CandlCobayaLikelihood:
            data_set_file: candl.data.SPT3G_2018_TTTEEE # data set or path to .yaml file
            lensing: False # Switch on for lensing likelihoods
            feedback: False # Switch on to request feedback from candl initialisation
            data_selection: None # Select a subset of the data set
            clear_internal_priors: True # Switch off to use candl internal priors

Only ``data_set_file`` is required, the other arguments are optional.
See :ref:`here <Data Selection>` for info on data selection.

*Note:* for the command-line interface, internal priors are ignored by default, i.e. it is assumed that cobaya handles these.
In contrast, for the interactive interface, candl-internal priors are applied.

MontePython
-------------------------------------------------

montepython, a bit of a pain but can be done.

BlackJAX
-------------------------------------------------

`BlackJAX <https://github.com/blackjax-devs/blackjax>`__ gradient-based samplers for JAX.
We show how to interface with BlackJAX and run NUTS chains in :ref:`the differentiability tutorial <Tutorials>`, but below is a quick example with the ACT DR4 likelihood using CosmoPower-JAX to obtain theory spectra.
First, we initialise the likelihood and the theory code:

.. code-block:: python

    import candl
    import candl.data
    import jax

    candl_like = candl.Like(candl.data.ACT_DR4_TTTEEE)
    cp_emulator_filenames = {"TT": "cmb_spt_TT_NN",
                             "TE": "cmb_spt_TE_PCAplusNN",
                             "EE": "cmb_spt_EE_NN"}
    theory_calc = candl.interface.CobayaTheoryCosmoPowerJAX(cp_emulator_filenames)
    pars_to_theory_specs = candl.interface.get_cobaya_pars_to_theory_specs_func(theory_calc)
    like = jax.jit(candl.tools.get_params_to_logl_func(candl_like, pars_to_theory_specs))

In the last line we obtain a function that moves from a dictionary of cosmological parameters to the log likelihood value.
For BlackJAX, it's easier to work with a normalised parameter vector as an input.
So next, we define the Planck 18 best-fit point as our reference and use the Fisher matrix (from this point) to normalise the parameters.

.. code-block:: python

    do that

Now we can follow the BlackJAX tutoriual to run NUTS chains.

.. code-block:: python

    do that


Optax
-------------------------------------------------

`Optax <https://github.com/google-deepmind/optax>`__ is a library for gradient-based optimisation written for JAX.
We show how to interface with Optax and run the ADAM minimiser in :ref:`the differentiability tutorial <Tutorials>`.