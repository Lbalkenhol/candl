candl - CosmoSIS interface
---------------------------

The files in this folder help you load any candl (https://github.com/Lbalkenhol/candl) likelihood into CosmoSIS.

Usage
---------------------------

In order run chains with e.g. the SPT-3G 2018 lensing likelihood, include the following block in your ``.ini`` file:

file = ./likelihood/candl/candl_cosmosis_interface.py ; Location of interface code - change depending on the location of your .ini file
data_set = 'candl.data.SPT3G_2018_Lens' ; Data set or path to .yaml file
variant = 'use_CMB' ; Select a variant of the data set if pointing to an index file
lensing = T ; Switch on for lensing likelihoods
feedback = T ; Switch on to request feedback from candl initialisation
data_selection = "..." ; Select a subset of the data set
clear_1d_internal_priors = T ; Switch off to use candl internal 1d priors
clear_nd_internal_priors = F ; Switch on to ignore candl internal higher dimensional priors. Careful: higher-dimensional priors are not implemented in CosmoSIS itself.
force_ignore_transformations = '' ; Backdoor if you want to ignore certain transformations in the data model.

Mofidy the information above as needed for other data sets. You can also consult the candl documentation for more information (https://candl.readthedocs.io/en/latest/).

Note that by default the 1-dimensional internal priors declared in candl's data set ``.yaml`` file are ignored, while the multi-dimensional priors are applied. If you want to modify this behaviour, set ``clear_1d_internal_priors`` and ``clear_nd_internal_priors``.

Credit
---------------------------

If you use this wrapper in your work please cite candl (https://arxiv.org/abs/2401.13433) as well as the relevant paper for all likelihoods you have accessed.

Authors
---------------------------

This wrapper was written by Y. Omori and L. Balkenhol, with help from J. Zuntz. If you have any questions or experience issues, please feel free to contact L. Balkenhol (lennart.balkenhol_at_iap.fr).
