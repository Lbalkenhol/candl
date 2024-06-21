# Changelog

## v1.3.0 (2024-06-21)

Tidied up Cobaya interface under the hood and various small improvements, including:

* Tidied up and unified Cobaya interfaces under the hood
    * __NOTE:__ Change in default behaviour: candl internal priors now ignored. Check docs for how to keep them.
* Improved speed of likelihoods when not using JAX
* Added option to return Fisher matrices (i.e. non-inverted) in `candl.tools.get_fisher_matrix`

## v1.2.0 (2024-04-26)

Improved flexibility of the interface and of lensing likelihoods:

* Fixed issue with ell-cropping lensing likelihoods
* Cobaya is now optional for all interface functions
* Improved flexibility of lensing likelihood theory code interfaces
* Improved flexibility of lensing likelihood sampler interfaces

## v1.1.0 (2024-03-20)

candl release v1.1.0. Several improvements, including:

* SPT-3G 2018 TT/TE/EE data set is now differentiable. Thanks to Marco Bonici for the pointer.
* CLASS theory wrapper is now more robust and can be passed a dictionary including nuisance parameter values.
* Added operation hints for transformations that make undoing them with candl.tools methods more robusts.
* Added version number to data set names.


## v1.0.1 (2024-02-06)

Initial release of candl.
