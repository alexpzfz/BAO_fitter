# BAO Fitter

This is (yet another) implementation of the BAO fitter based on Beutler et al. (2017).
It provides a straightforward way to load data, set the model and run the fits. What you see is what you get. This is not an attempt to replace any of the existing codes, but rather an option for those who require a quick set-up.

Because BAO fits should be fun and simple.

The fitter works in configuration and Fourier space, but it does not currenlty support a window function in Fourier space (**coming soon**).

It currently uses Zeus (https://github.com/minaskar/zeus) as the default sampler, but other samplers should be relatively simple to add.

#### Requirements:
- Numpy
- Scipy

Optional:
- Zeus
- mpi4py
- getdist

#### Acknowledgements:
Thanks to Mariana Vargas Magaña and Sebastien Fromenteau.

The object-oriented approach has been inspired by:
- BARRY: https://github.com/Samreay/Barry/
- Cosmoprimo: https://github.com/cosmodesi/cosmoprimo 