.. DiffiLQRax Documentation master file, created by
   sphinx-quickstart on Thu Dec 19 23:21:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: _static/images/diffilqrax_logo_dm.png
   :align: center
   :width: 420px
   :class: only-dark

.. image:: _static/images/diffilqrax_logo.png
   :align: center
   :width: 420px
   :class: only-light



.. rst-class:: heading-center


Welcome to DiffiLQRax!
====================================

DiffiLQRax is an open source Python package that provides a differentiable implementation of the iterative Linear Quadratic Regulator (iLQR) algorithm using the JAX library. 


Please support the development of DiffiLQRax by starring and/or watching the project on Github!


Quick Examples
--------------

LQR Examples
^^^^^^^^^^^^

.. grid:: 2

   .. grid-item-card::  LQR optimal control of integrator dynamics
      :link: ./examples/lqr_integrator
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: _static/nb_thumbnails/lqr_integrator.png
         :alt: Thumbnail for LQR solution
         :class: gallery-thumbnail

   .. grid-item-card::  LQR tracking optimal control of integrator dynamics
      :link: ./examples/lqr_tracking_integrator
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: _static/nb_thumbnails/lqr_tracking_integrator.png
         :alt: Thumbnail for LQR tracking solution
         :class: gallery-thumbnail

iLQR Examples
^^^^^^^^^^^^^

.. grid:: 3

   .. grid-item-card::  iLQR solver for simple pendulum
      :link: ./examples/simple_pendulum
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: _static/nb_thumbnails/simple_pendulum.png
         :alt: Thumbnail for iLQR solution for pendulum
         :class: gallery-thumbnail

   .. grid-item-card::  iLQR Optimal Control compared with Gradient Descent
      :link: ./examples/ilqr_vs_backpropagation
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: _static/nb_thumbnails/ilqr_vs_backpropagation.png
         :alt: Thumbnail for iLQR vs gradient descent
         :class: gallery-thumbnail

   .. grid-item-card::  iLQR solver Implicit vs. Direct gradients
      :link: ./examples/gradients_through_ilqr
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: _static/nb_thumbnails/gradients_through_ilqr.png
         :alt: Thumbnail for iLQR solution and implicit-direct gradients
         :class: gallery-thumbnail
    
Installation
------------

To get started with this code, clone the repository and install the required dependencies. Then, you can run the main script to see the iLQR in action.

.. code-block:: bash

   git clone git@github.com:ThomasMullen/diffilqrax.git
   cd diffilqrax
   python -m build
   pip install -e .

or, you can import from pip install

.. code-block:: bash

   pip install diffilqrax



Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   userguide/getting_started
   examples/index
   userguide/design_principles
   about/lqr_theory
   api/modules


Acknowledgements
================
This project has received funding from the European Union’s Horizon 2020 research 
and innovation programme under the Marie Skłodowska-Curie grant agreement #813457.