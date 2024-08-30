Likelihood Code
=================================================

.. toctree::
   :titlesonly:
   :maxdepth: 1
   :hidden:

   like_data_structure

The likelihood is a function that takes a set of CMB spectra and parameter values returns a log-likelihood value.
For the default functional form the likelihood is a multivariate Gaussian:

.. math::
   -2\log{\mathcal{L}(\theta)} = \left\{\hat{D} - \mathcal{T}\left[D^{\mathrm{CMB}}(\theta), \theta \right]\right\}^T \mathcal{C}^{-1} \left\{\hat{D} - \mathcal{T}\left[D^{\mathrm{CMB}}(\theta), \theta \right]\right\},

where :math:`\hat{D}` is the data vector, :math:`\mathcal{T}` is the data model, :math:`D^{\mathrm{CMB}}(\theta)` is the model CMB power spectrum, :math:`\mathcal{C}` is the covariance matrix, and :math:`\theta` are the parameters.
The data model is comprised of a series of successively applied transformations:

.. math::
   \mathcal{T}\left[D^{\mathrm{CMB}}(\theta), \theta \right] = \mathcal{T}_N(\mathcal{T}_{N-1}(\dots \mathcal{T}_{0}(D^{\mathrm{CMB}}(\theta), \dots), \theta) \theta).

These transformations adjust the power spectrum supplied by the theory code to account for foregrounds, calibration, super-sample lensing or any other effect.
The transformations are applied in the order specified in the data set info file.
Programmatically, the data band powers model spectrum are long vectors of concatenated spectra according to the order defined in the data set info file.

.. note::

    The likelihood operates in :math:`D_\ell` space, i.e. on :math:`C_\ell \ell (\ell + 1) / (2 \pi)`, in units of :math:`\mu K_{\mathrm{CMB}}^2`.
    Theory spectra start at :math:`\ell=2`.

Primary CMB Likelihood
-----------------------------

The likelihood for primary CMB data proceeds in the following steps:

#. Get model spectra:

   a. Construct a long vector of CMB-only spectra (matching the order of the data vector).

   b. Loop over the transformations in the data model and apply them one by one to the model vector.

#. Bin the model spectra using the window functions.

#. Take the difference between data and model band powers and calculate the log likelihood value.

#. Calculate and add the prior contribution.


Lensing CMB Likelihood
-----------------------------

For the lensing likelihood, the model spectra are binned prior to applying the transformations.
Due to the nature of transformations that are typically applied to lensing spectra (e.g., matrix multiplications for response function corrections), is more efficient.
The order is:

#. Construct a long vector of CMB-only spectra (matching the order of the data vector).

#. Bin the CMB-only spectra using the window functions.

#. Loop over the transformations in the data model and apply them one by one to the model vector.

#. Take the difference between data and model band powers and calculate the log likelihood value.

#. Calculate and add the prior contribution.
