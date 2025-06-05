# Changelog

## v2.0.2 (2025-06-05)

Removed more occurences of deprecated candl.data.

## v2.0.1 (2025-06-04)

Fixed bug in cobaya wrapper when loading clipy likelihoods.

## v2.0.0 (2025-06-03)

Separated code and data. To download and install data sets, go to the `candl_data` repository (https://github.com/Lbalkenhol/candl_data).

## v1.8.2 (2025-06-02)

Fixed bug in file template file reading.

## v1.8.1 (2025-05-30)

Made cobaya wrapper more flexible.

## v1.8.0 (2025-05-29)

* Updated ACT DR6 lensing likelihood
* Added flexibility to read in files for transformations
* Added npy io option

## v1.7.0 (2025-03-24)

Added ACT DR6 CMB-only likelihood.

## v1.6.0 (2025-01-31)

Added support for clipy Planck likelihoods, which now ship with candl by default, including a tutorial notebook. Various tweaks to improve flexibility and usability of the code.

## v1.5.2 (2024-12-03)

Added foreground-marginalised 'lite' version of the SPT-3G 2018 TT/TE/EE likelihood and a notebook showing its construction.

## v1.5.1 (2024-09-06)

Compatibility with JAX version 0.4.31 (tested with TF 2.17.0).

## v1.5.0 (2024-08-30)

Added index files to help navigate between variants of a likelihood, plus few other small fixes and changes. Restructured the docs, pulling tutorials and usage tips apart.

## v1.4.0 (2024-07-15)

A few fixes to add more flexibility.

* Can now supply full file paths to CosmoPower models in interface
* Made behaviour of Cobaya interface more explicit
* Improved stability of data selection
* Added option to apply Hartlap correction to inverse covariance

## v1.3.0 (2024-06-21)

Tidied up Cobaya interface under the hood and various small improvements, including:

* Tidied up and unified Cobaya interfaces under the hood
    * __NOTE:__ Change in default behaviour: candl internal priors now ignored. Check docs for how to keep them
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
