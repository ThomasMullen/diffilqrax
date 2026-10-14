.. _design_principles:

Design Principles
=================

Each problem class, LQR or iLQR, has a sequential and a parallel solver. Each solver also has a
differentiable wrapper with implicit gradients.

+----------------+--------------------------------+------------------------------------------------+----------------+-----------------------------+
| Solver type    | Module                         | Entry point                                    | Suited to      | Arguments                   |
+================+================================+================================================+================+=============================+
| Sequential     | ``diffilqrax.lqr``             | :func:`~diffilqrax.lqr.solve_lqr`              | CPU            | ``LQRParams``               |
|                +--------------------------------+------------------------------------------------+----------------+-----------------------------+
|                | ``diffilqrax.ilqr``            | :func:`~diffilqrax.ilqr.ilqr_solver`           | CPU            | ``System, iLQRParams``      |
+----------------+--------------------------------+------------------------------------------------+----------------+-----------------------------+
| Parallel       | ``diffilqrax.plqr``            | :func:`~diffilqrax.plqr.solve_plqr`            | GPU            | ``LQRParams``               |
|                +--------------------------------+------------------------------------------------+----------------+-----------------------------+
|                | ``diffilqrax.parallel_ilqr``   | :func:`~diffilqrax.parallel_ilqr.pilqr_solver` | GPU            | ``System, iLQRParams``      |
+----------------+--------------------------------+------------------------------------------------+----------------+-----------------------------+
| Differentiable | ``diffilqrax.diff_lqr``        | :func:`~diffilqrax.diff_lqr.dlqr`              | CPU            | ``ModelDims, LQRParams``    |
|                +--------------------------------+------------------------------------------------+----------------+-----------------------------+
|                | ``diffilqrax.diff_ilqr``       | :func:`~diffilqrax.diff_ilqr.dilqr`            | CPU            | ``System, iLQRParams``      |
|                +--------------------------------+------------------------------------------------+----------------+-----------------------------+
|                | ``diffilqrax.parallel_dlqr``   | :func:`~diffilqrax.parallel_dlqr.pdlqr`        | GPU            | ``ModelDims, LQRParams``    |
|                +--------------------------------+------------------------------------------------+----------------+-----------------------------+
|                | ``diffilqrax.parallel_dilqr``  | :func:`~diffilqrax.parallel_dilqr.pdilqr`      | GPU            | ``System, iLQRParams``      |
+----------------+--------------------------------+------------------------------------------------+----------------+-----------------------------+

The layout of the modules are hierarchically structure outlined in the schematic below:

.. image:: ../_static/images/diffilqrax_design_principle.png
   :align: center
   :class: only-dark

.. image:: ../_static/images/diffilqrax_design_principle.png
   :align: center
   :class: only-light
