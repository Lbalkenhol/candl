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

``CMBlite_using_autodiff.ipynb``
--------------------------------------------

This notebook accompagnies `Balkenhol 2024 <https://arxiv.org/abs/2412.00826>`__ and shows how to construct foreground-marginalised CMBlite likelihoods with automatic differentiation using the SPT-3G 2018 TT/TE/EE data set as an example.
