.. _Changelog:

What's new
##########

.. warning::
    EntroPy is now **DEPRECATED**. Please use the `AntroPy package <https://github.com/raphaelvallat/antropy>`_ instead!


v0.1.4 (March 2021)
-------------------

a. Updated :py:func:`entropy.lziv_complexity` using integer arrays with `Numba` in `this pull request <https://github.com/raphaelvallat/entropy/pull/16>`_ 
b. :py:func:`entropy.lziv_complexity` now maps strings to UTF-8 integer representations, and uses `np.bincount` for computing base

v0.1.3 (March 2021)
-------------------

a. Added the :py:func:`entropy.num_zerocross` function to calculate the (normalized) number of zero-crossings on N-D data.
b. Added the :py:func:`entropy.hjorth_params` function to calculate the mobility and complexity Hjorth parameters on N-D data.
c. Add support for N-D data in :py:func:`entropy.spectral_entropy`, :py:func:`entropy.petrosian_fd` and :py:func:`entropy.katz_fd`.
d. Use the `stochastic <https://github.com/crflynn/stochastic>`_ package to generate stochastic time-series.

v0.1.2 (May 2020)
-----------------

a. :py:func:`entropy.lziv_complexity` now works with non-binary sequence (e.g. "12345" or "Hello World!")
b. The average fluctuations in :py:func:`entropy.detrended_fluctuation` is now calculated using the root mean square instead of a simple arithmetic. For more details, please refer to `this GitHub issue <https://github.com/neuropsychology/NeuroKit/issues/206>`_.
c. Updated flake8

v0.1.1 (November 2019)
----------------------

a. Added Lempel-Ziv complexity (:py:func:`entropy.lziv_complexity`) for binary sequence.

v0.1.0 (October 2018)
---------------------

Initial release.

a. Permutation entropy
b. Spectral entropy
c. Singular value decomposition entropy
d. Approximate entropy
e. Sample entropy
f. Petrosian Fractal Dimension
g. Katz Fractal Dimension
h. Higuchi Fractal Dimension
i. Detrended fluctuation analysis
