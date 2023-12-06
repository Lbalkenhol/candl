.. _tests_api:

``candl.tests``
====================================

Tests are supplied for each released likelihood to verify that they operate correctly.
A test is comprised of a ``.yaml`` file with the following contents:

.. code-block:: yaml

   data_set_file: "path_to_the_data_set_file.yaml"
   lensing: False # set True for lensing likelihoods
   test_spectrum: "tests/path_to_test_spectrum.txt" # spectrum to test
   param_values: {"y": 1.0} # nuisance parameter values
   test_chisq: 300.00 # expected chi-squared value

The file pointed to by ``test_spectrum`` is a text file with the following columns: ``ell, TT, TE, EE, BB, PP, KK``, starting at :math:`\ell = 2`.
Instead of ``test_chisq``, you can also supply ``test_logl``, which will test the log-likelihood value.
A test is considered to pass if the returned chi-squared or log-likelihood value is within 0.1% of the expected value.

.. autofunction:: candl.tests.run_test

.. autofunction:: candl.tests.run_all_tests
