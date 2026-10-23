.. _getting_started:

Getting started
===============

Installation
------------

diffilqrax needs Python 3.10 or later and depends only on ``jax``, ``numpy`` and ``jaxopt``.
To install from a local clone with ``pip``:

.. code-block:: bash

   git clone git@github.com:ThomasMullen/diffilqrax.git
   cd diffilqrax
   pip install -e .              # library only
   pip install -e ".[dev]"       # + test and lint tools
   pip install -e ".[doc]"       # + documentation tools

Or with `uv <https://docs.astral.sh/uv/>`_, which uses the committed ``uv.lock``:

.. code-block:: bash

   uv sync --extra dev            # creates .venv with the locked dependencies
   uv run pytest tests/ -m "not slow"

For GPU support, install the matching ``jax`` build first, as described in the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.

.. note::

   Importing a diffilqrax solver turns on JAX double precision (``jax_enable_x64``). The
   default tolerances, for example ``kkt_tol=1e-10``, assume float64.

Core types
----------

All problem data is held in the small ``NamedTuple`` containers defined in :mod:`diffilqrax.typs`:

``ModelDims(n, m, horizon, dt)``
    State dimension, input dimension, number of time steps :math:`T`, and the time step.

``LQR(A, B, a, Q, q, R, r, S, Qf, qf)``
    A time-varying LQR problem. The time-varying terms are stacked along a leading axis of length
    :math:`T`: ``A`` has shape ``(T, n, n)``, ``B`` has shape ``(T, n, m)`` and ``S`` has shape
    ``(T, n, m)``. The terminal terms ``Qf`` and ``qf`` have no time axis.

``LQRParams(x0, lqr)``
    Initial state and the ``LQR`` problem.

``System(cost, costf, dynamics, dims)``
    A non-linear problem. Every function takes the parameters ``theta`` as its last argument:
    ``cost(t, x, u, theta)``, ``costf(x, theta)`` and ``dynamics(t, x, u, theta)``, which returns
    :math:`x_{t+1}`. Derivatives are taken with JAX automatic differentiation.

``iLQRParams(x0, theta)``
    Initial state and an arbitrary pytree of parameters. Gradients are taken with respect to this
    pytree.

Solving an LQR problem
----------------------

Consider optimal control for a double integrator driving point mass to rest state.
The linear state dynamics are

.. math::
    :label: dynamics

    x_{t+1} = A x_t + B u_t, \qquad
    A = \begin{pmatrix} 1 & \delta t \\ 0 & 1 \end{pmatrix}, \quad
    B = \begin{pmatrix} \tfrac12 \delta t^2 \\ \delta t \end{pmatrix}.

The solver is searching for a control sequence that minimises quadratic cost

.. math::
    \min_{u_{0:T-1}} \; J = \tfrac{1}{2} x_T^\top Q_f x_T
   + \tfrac{1}{2} \sum_{t=0}^{T-1} \left( x_t^\top Q x_t + r\, u_t^2 \right), \\
    Q = \begin{pmatrix} 1 & \delta t \\ 0 & 1 \end{pmatrix}, \quad
    Q_f = \begin{pmatrix} 10 & \delta t \\ 0 & 10 \end{pmatrix}, \quad
    R = \begin{pmatrix} 0.1 & \delta t \\ 0 & 0.1 \end{pmatrix}, \quad

subject to :eq:`dynamics`, with :math:`Q, Q_f \succeq 0` and :math:`r > 0`.

The solution is a linear feedback law :math:`u_t = -K_t x_t`, where the gains
follow from the backward Riccati recursion, starting at :math:`V_T = Q_f`:

.. math::

   K_t &= \left( r + B_d^\top V_{t+1} B_d \right)^{-1} B_d^\top V_{t+1} A_d \\
   V_t &= Q + A_d^\top V_{t+1} \left( A_d - B_d K_t \right)


.. code-block:: python

    import jax.numpy as jnp
    from diffilqrax.typs import LQR, LQRParams, ModelDims
    from diffilqrax.lqr import solve_lqr

    dims = ModelDims(n=2, m=1, horizon=50, dt=0.1)
    T, n, m = dims.horizon, dims.n, dims.m

    A = jnp.array([[1.0, dims.dt], [0.0, 1.0]])
    B = jnp.array([[0.0], [dims.dt]])
    lqr = LQR(
        A=jnp.tile(A, (T, 1, 1)), a=jnp.zeros((T, n)),
        B=jnp.tile(B, (T, 1, 1)),
        Q=jnp.tile(jnp.eye(n), (T, 1, 1)), q=jnp.zeros((T, n)),
        S=jnp.zeros((T, n, m)),
        R=jnp.tile(0.1 * jnp.eye(m), (T, 1, 1)), r=jnp.zeros((T, m)),
        Qf=10.0 * jnp.eye(n), qf=jnp.zeros(n),
    )
    params = LQRParams(x0=jnp.array([1.0, 0.0]), lqr=lqr)

    Xs, Us, lambdas = solve_lqr(params) # shapes (T+1, n), (T, m), (T+1, n)

:func:`diffilqrax.plqr.solve_plqr` takes the same arguments and returns the same solution, computed
with an associative scan.

Solving a non-linear problem with iLQR
--------------------------------------

This example swings a pendulum up to :math:`\theta = \pi`:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from diffilqrax.typs import System, iLQRParams, PendulumParams
    from diffilqrax.ilqr import ilqr_solver

    DAMPING, U_MAX = 0.1, 3.0
    dims = ModelDims(n=2, m=1, horizon=100, dt=0.05)

    def dynamics(t, x, u, theta):
        """Euler step; x = [angle, angular velocity], angle 0 is hanging down"""
        acc = -theta.g / theta.l * jnp.sin(x[0]) - DAMPING * x[1] + u[0] / (theta.m * theta.l**2)
        return x + DT * jnp.stack([x[1], acc])

    def cost(t, x, u, theta):
        """Small running cost on angle, velocity and torque, plus a smooth barrier at |u| = U_MAX"""
        barrier = jax.nn.softplus(5.0 * (u - U_MAX)) ** 2 + jax.nn.softplus(5.0 * (-u - U_MAX)) ** 2
        return 0.1 * (1.0 + jnp.cos(x[0])) + 0.01 * x[1] ** 2 + 0.01 * jnp.sum(u**2) + jnp.sum(barrier)

    def costf(x, theta):
        """Be upright (angle ±pi) and still at the final time"""
        return 50.0 * (1.0 + jnp.cos(x[0])) + 5.0 * x[1] ** 2

    # Construct problem
    model = System(cost, costf, dynamics, dims)
    params = iLQRParams(x0=jnp.array([0.1, 0.0]), theta=PendulumParams(m=1.0, l=1.0, g=9.8))
    Us_init = 0.5 * jr.normal(jr.PRNGKey(0), (dims.horizon, dims.u))
    
    # Solve problem
    (Xs, Us, lambdas), total_cost, cost_log = ilqr_solver(model, params, Us_init, max_iter=300)
    

The main options are:

- ``kkt_tol``: the solver stops when :math:`\max_t |\partial L / \partial u_t|` is below this value.
- ``max_iter``: the maximum number of outer iterations.
- line-search keyword arguments for :func:`~diffilqrax.ilqr.armijo_linesearch`: ``tau``, ``beta``,
  ``alpha_min`` and ``max_iter_linesearch``.

Regularisation of the input Hessian is adaptive, so it needs no tuning.

Differentiating through the solver
----------------------------------

:func:`diffilqrax.diff_ilqr.dilqr` solves the same problem and returns
:math:`\tau = [x_t, u_t]` as an array of shape ``(T+1, n+m)``. It defines a custom VJP, so
``jax.grad`` gives exact implicit gradients with respect to ``x0`` and every leaf of ``theta``.
JAX does not backpropagate through the solver iterations.

.. code-block:: python

    import jax
    from diffilqrax.diff_ilqr import dilqr

    def loss(theta):
        tau = dilqr(model, iLQRParams(x0=jnp.zeros(2), theta=theta), Us_init, max_iter=200, kkt_tol=1e-6)
        return jnp.sum(tau[:, dims.n:] ** 2)   # control effort of the optimal solution

    grads = jax.grad(loss)(params.theta)       # a PendulumParams of gradients

For LQR problems, :func:`diffilqrax.diff_lqr.dlqr` does the same job: ``dlqr(dims, params, None)``.

Parallel solvers
----------------

:func:`diffilqrax.parallel_ilqr.pilqr_solver` and :func:`diffilqrax.parallel_dilqr.pdilqr` are
drop-in replacements for ``ilqr_solver`` and ``dilqr``. They give the same solution, but the
Riccati recursion becomes an associative scan:

.. code-block:: python

    from diffilqrax.parallel_ilqr import pilqr_solver

    (Xs, Us, _), J, _ = pilqr_solver(model, params, Us_init, max_iter=200, kkt_tol=1e-6)

With ``rollout="parallel"``, the non-linear rollout is also replaced, by multiple shooting, so each
iteration has :math:`O(\log T)` depth. This is worthwhile on a GPU with long horizons. On a CPU the
sequential solvers are usually faster. See :doc:`../examples/parallel_solvers` for a comparison.

Next steps
----------

- :doc:`../examples/index` has worked notebooks, each ending with a *How to adapt* section.
- :doc:`design_principles` gives an overview of the modules.
- ``examples/`` in the repository has standalone scripts, such as the pendulum swing-up and a
  4000-step tracking problem.

Building the documentation locally
----------------------------------

.. code-block:: bash

   pip install -e ".[doc]"        # also needs the pandoc executable, e.g. brew install pandoc
   make -C docs html              # or: sphinx-build -b html docs/source docs/build/html
   open docs/build/html/index.html
