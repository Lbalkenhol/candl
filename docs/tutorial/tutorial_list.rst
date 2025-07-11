Tutorials
=================================================

Below, can find information on the provided tutorials that are available in the ``notebooks/`` folder on the git repo.
The ``traditional_tutorial.ipynb`` and ``differentiable_tutorial.ipynb`` are intended for you to run on your machine and can serve as jumping off points for your own work.
The ``SPT_ACT_summer_school_2024_candl.ipynb`` is a longer tutorial with exercises that was designed for the joint SPT and ACT analysis summer school 2024 and takes a more pedagological angle.
You can run it on your own machine or on google colab.


``traditional_tutorial.ipynb``
--------------------------------------------

This notebook shows how traditional inference tasks are accomplished. In particular:

* Initialising the likelihood and accessing the data (band powers, covariance, etc.)
* Interfacing the likelihood with CAMB and calculating the :math:`\chi^2` for a given spectrum
* Interfacing the likelihood with Cobaya and running an MCMC chain

This tutorial uses some optional packages.
Make sure you have Cobaya, getdist, and CAMB installed in order to run the whole notebook.

``differentiable_tutorial.ipynb``
--------------------------------------------

This notebook shows different aspects relying on the differentiability of the likelihood. In particular:

* Initialising the likelihood and accessing the data (band powers, covariance, etc.)
* Running gradient-based minimisers
* Interfacing the likelihood with Optax
* Running NUTS chains by interfacing the likelihood with BlackJAX

This tutorial uses some optional packages.
Make sure you have Optax, BlackJAX, getdist, and CosmoPower-JAX installed in order to run the whole notebook.
You also need to have some emulator models for CosmoPower-JAX; we recommend the SPT high-accuracy models available `here <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/SPT_high_accuracy>`_.

``SPT_ACT_summer_school_2024_candl.ipynb``
--------------------------------------------

This notebook was designed for the joint SPT and ACT analysis summer school 2024.
This is not a pure click-through tutorial, but features some exercises.
It covers in three parts:

* Part I candl Basics: This part will run you through the basics of using candl: how to initialise a likelihood, access different data products, understand the data model, and evaluate the likelihood. For this part of the notebook you will be using the SPT-3G 2018 TT/TE/EE data set.
* Part II Building a Differentiable Pipeline, Calculating Derivatives: in this part you will build a differentiable pipeline from cosmological parameters all the way to the likelihood value. We will look at two useful applications using the ACT DR4 TT/TE/EE data set.
* Part III Gradient-Powered Likelihood Exploration: in this part you will find the best-fit point of the ACT DR4 TT/TE/EE data set using traditional and gradient-powered methods.

You can run this tutorial locally or on `google colab <https://github.com/Lbalkenhol/candl/blob/main/notebooks/SPT_ACT_summer_school_2024/SPT_ACT_summer_school_2024_candl_colab.ipynb>`_.

``SPT3G_D1_TnE_tutorial.ipynb``
--------------------------------------------

`This notebook <https://github.com/SouthPoleTelescope/spt_candl_data/blob/main/tutorial_notebooks/SPT3G_D1_TnE_tutorial.ipynb>`_ from the SPT-3G collaboration shows how to interact with the SPT-3G D1 T&E data using candl in two parts.
The first part shows you how to initialize the multifrequency likelihood, evaluate it, visualize the data, and helps you understand the data model.
The second part uses the `SPTlite` likelihood and leverages the differentiability of `candl`. This part shows how to translate biases from the band-power level to the parameter-level and how to perform gradient-based minimization and sampling.

``CMBlite_using_autodiff.ipynb``
--------------------------------------------

This notebook accompagnies `Balkenhol 2024 <https://arxiv.org/abs/2412.00826>`__ and shows how to construct foreground-marginalised CMBlite likelihoods with automatic differentiation using the SPT-3G 2018 TT/TE/EE data set as an example.

``clipy_x_candl_tutorial.ipynb``
--------------------------------------------

This notebook shows how to access ``clipy`` Planck likelihoods through their candl wrapper. In particular it covers:

* Initialising Planck likelihoods
* Taking derivatives of Planck likelihoods
* Interfacing the clipy with Cobaya through candl and running MCMC chains.

This tutorial uses some optional packages.
Make sure you have Cobaya and CosmoPower-JAX installed in order to run the whole notebook.
You also need to download some `Planck data files <https://pla.esac.esa.int/pla>`_ and have some emulator models for CosmoPower-JAX; we recommend the SPT high-accuracy models available `here <https://github.com/alessiospuriomancini/cosmopower/tree/main/cosmopower/trained_models/SPT_high_accuracy>`_.
