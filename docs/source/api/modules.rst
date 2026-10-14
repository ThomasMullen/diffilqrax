API Reference
=============

Every solver comes in a sequential form, which uses a Riccati recursion and suits CPUs, and a
parallel form, which uses an associative scan with :math:`O(\log T)` depth and suits GPUs on long
horizons. Each also has a differentiable wrapper. The wrapper returns
:math:`\tau = [x_t, u_t]` and has implicit (adjoint-LQR) gradients with respect to the problem
parameters.

Solvers
-------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: lqr
      :link: modules/diffilqrax_lqr
      :link-type: doc

      Sequential LQR: :func:`~diffilqrax.lqr.solve_lqr`.

   .. grid-item-card:: ilqr
      :link: modules/diffilqrax_ilqr
      :link-type: doc

      Sequential iLQR: :func:`~diffilqrax.ilqr.ilqr_solver`.

   .. grid-item-card:: plqr
      :link: modules/diffilqrax_plqr
      :link-type: doc

      Parallel LQR: :func:`~diffilqrax.plqr.solve_plqr`.

   .. grid-item-card:: parallel_ilqr
      :link: modules/diffilqrax_pilqr
      :link-type: doc

      Parallel iLQR: :func:`~diffilqrax.parallel_ilqr.pilqr_solver`.

Differentiable solvers
----------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: diff_lqr
      :link: modules/diffilqrax_dlqr
      :link-type: doc

      :func:`~diffilqrax.diff_lqr.dlqr`

   .. grid-item-card:: diff_ilqr
      :link: modules/diffilqrax_dilqr
      :link-type: doc

      :func:`~diffilqrax.diff_ilqr.dilqr`

   .. grid-item-card:: parallel_dlqr
      :link: modules/diffilqrax_pdlqr
      :link-type: doc

      :func:`~diffilqrax.parallel_dlqr.pdlqr`

   .. grid-item-card:: parallel_dilqr
      :link: modules/diffilqrax_pdilqr
      :link-type: doc

      :func:`~diffilqrax.parallel_dilqr.pdilqr`

Types and utilities
-------------------

.. grid:: 1 2 2 2
   :gutter: 2

   .. grid-item-card:: typs
      :link: modules/diffilqrax_typs
      :link-type: doc

      ``ModelDims``, ``System``, ``LQR``, ``LQRParams``, ``iLQRParams`` and more.

   .. grid-item-card:: utils
      :link: modules/diffilqrax_utils
      :link-type: doc

      Key splitting, random stable dynamics and linearisation helpers.

.. toctree::
   :hidden:

   modules/diffilqrax_lqr
   modules/diffilqrax_ilqr
   modules/diffilqrax_plqr
   modules/diffilqrax_pilqr
   modules/diffilqrax_dlqr
   modules/diffilqrax_dilqr
   modules/diffilqrax_pdlqr
   modules/diffilqrax_pdilqr
   modules/diffilqrax_typs
   modules/diffilqrax_utils
