.. _getting_started:
Getting started
===============



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

Dependencies
^^^^^^^^^^^^
diffiLQRax requires Python 3.10 or later. 
It is built on top of JAX so the minimum requirements are:

.. code-block:: bash

    build
    numpy
    jax
    jaxopt
    chex
    pytest
    matplotlib


Quickstart
----------

Here is a simple example of how to use the LQR and iLQR solvers:

.. code-block:: Python

    import jax.numpy as jnp
    from diffilqrax.typs import LQRParams
    from diffilqrax import lqr

    # Set-up LQR params e.g. cost & dynamics matrices
    lqr_params = LQRParams(x_init=jnp.zeros(3), lqr=mat_params)

    # Solve LQR problem
    opt_xs, opt_us, opt_adjoints = lqr.solve(lqr_params)

To use the differentiable solver, import :code:`diff_lqr` instead of :code:`lqr`. 
To see the different available solvers, refer to the :doc:`design_principles`.

.. code-block:: Python

    import jax.numpy as jnp
    import jax.random as jr
    from diffilqrax import ilqr
    from diffilqrax.typs import iLQRParams, Theta, ModelDims, System
    from diffilqrax.utils import initialise_stable_dynamics, keygen
    
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 5)

    # Define model dimensionality
    dims = ModelDims(8, 2, 100, dt=0.1)

    # Define model parameters
    Uh = initialise_stable_dynamics(next(skeys), dims.n, dims.horizon, 0.6)[0]
    Wh = jr.normal(next(skeys), (dims.n, dims.m))
    theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(dims.n), Q=jnp.eye(dims.n))
    params = iLQRParams(x0=jr.normal(next(skeys), dims.n), theta=theta)
    
    # Initialise control sequence
    Us = jnp.zeros((dims.horizon, dims.m))

    # Define linesearch hyper-parameters
    ls_kwargs = {
        "beta":0.8,
        "max_iter_linesearch":16,
        "tol":1e0,
        "alpha_min":0.0001,
        }

    # Set-up problem
    def cost(t, x, u, theta):
        return jnp.sum(x**2) + jnp.sum(u**2)

    def costf(x, theta):
        return jnp.sum(x**2)

    def dynamics(t, x, u, theta):
        return jnp.tanh(theta.Uh @ x + theta.Wh @ u)

    model = System(cost, costf, dynamics, dims)

    # Solve problem
    (opt_xs, opt_us, opt_adjoints), total_cost = ilqr.ilqr_solver(params, model, Us, **ls_kwargs)