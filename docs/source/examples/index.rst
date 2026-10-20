Tutorials
=========

Worked examples, from a single LQR solve to differentiating through iLQR. Each notebook explains
the problem, solves it, checks the result and ends with a **How to adapt** section. The
repository's ``examples/`` folder has standalone script versions.

Linear Quadratic Regulator (LQR)
--------------------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card::  LQR: bringing a double integrator to rest
      :link: ./lqr_integrator
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/lqr_integrator.png
         :alt: LQR regulation
         :class: gallery-thumbnail

   .. grid-item-card::  LQR tracking of a reference trajectory
      :link: ./lqr_tracking_integrator
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/lqr_tracking_integrator.png
         :alt: LQR tracking
         :class: gallery-thumbnail


Iterative LQR (iLQR)
--------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card::  iLQR: pendulum swing-up
      :link: ./simple_pendulum
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/simple_pendulum.png
         :alt: pendulum swing-up
         :class: gallery-thumbnail

   .. grid-item-card::  iLQR compared with gradient descent
      :link: ./ilqr_vs_backpropagation
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/ilqr_vs_backpropagation.png
         :alt: iLQR vs gradient descent
         :class: gallery-thumbnail

   .. grid-item-card::  Implicit gradients through iLQR
      :link: ./gradients_through_ilqr
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/gradients_through_ilqr.png
         :alt: implicit vs direct gradients
         :class: gallery-thumbnail

   .. grid-item-card::  Parallel (associative-scan) solvers
      :link: ./parallel_solvers
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/parallel_solvers.png
         :alt: parallel solvers
         :class: gallery-thumbnail

   .. grid-item-card::  iLQR inference: reconstructing a zebrafish tail bout
      :link: ./zebrafish_tail_reconstruction
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/zebrafish_tail_reconstruction.png
         :alt: zebrafish tail reconstruction
         :class: gallery-thumbnail

   .. grid-item-card::  Batched iLQR: many tracking problems at once
      :link: ./batched_ilqr
      :link-type: doc
      :shadow: none
      :class-card: example-gallery

      .. image:: ../_static/nb_thumbnails/batched_ilqr.png
         :alt: batched iLQR scaling
         :class: gallery-thumbnail



.. toctree::
   :hidden:

   lqr_integrator
   lqr_tracking_integrator
   simple_pendulum
   ilqr_vs_backpropagation
   gradients_through_ilqr
   parallel_solvers
   zebrafish_tail_reconstruction
   batched_ilqr
