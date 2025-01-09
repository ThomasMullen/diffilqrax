Tutorials
=========

This section contains a series of tutorials to help you use the diffiLQRax package for optimal control.

Linear Quadratic Regulator (LQR) Problem
----------------------------------------

Demos
^^^^^

.. grid:: 2

   .. grid-item-card::  LQR optimal control of integrator dynamics
      :link: ./lqr_integrator
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/lqr_integrator.png
         :alt: Thumbnail for LQR solution
         :class: gallery-thumbnail

   .. grid-item-card::  LQR tracking optimal control of integrator dynamics
      :link: ./lqr_tracking_integrator
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/lqr_tracking_integrator.png
         :alt: Thumbnail for LQR tracking solution
         :class: gallery-thumbnail


Iterative Linear Quadratic Regulator (iLQR) Problem
---------------------------------------------------

Demos
^^^^^

.. grid:: 3

   .. grid-item-card::  iLQR solver for simple pendulum
      :link: ./simple_pendulum
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/simple_pendulum.png
         :alt: Thumbnail for iLQR solution for pendulum
         :class: gallery-thumbnail

   .. grid-item-card::  iLQR Optimal Control compared with Gradient Descent
      :link: ./ilqr_vs_backpropagation
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/ilqr_vs_backpropagation.png
         :alt: Thumbnail for iLQR vs gradient descent
         :class: gallery-thumbnail

   .. grid-item-card::  iLQR solver Implicit vs. Direct gradients
      :link: ./gradients_through_ilqr
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/gradients_through_ilqr.png
         :alt: Thumbnail for iLQR solution and implicit-direct gradients
         :class: gallery-thumbnail
    

.. toctree::
   :hidden:

   lqr_integrator
   lqr_tracking_integrator
   simple_pendulum
   ilqr_vs_backpropagation
   gradients_through_ilqr