================
DiffiLQRax
================

.. image:: https://github.com/ThomasMullen/diffilqrax/actions/workflows/pylint.yml/badge.svg
   :align: left

.. image:: https://github.com/ThomasMullen/diffilqrax/actions/workflows/python-package.yml/badge.svg
   :align: center

.. image:: https://github.com/ThomasMullen/diffilqrax/actions/workflows/python-publish.yml/badge.svg
   :align: right

This repository contains an implementation of the iterative Linear Quadratic Regulator (iLQR) using the JAX library. The iLQR is a powerful algorithm used for optimal control, and this implementation is designed to be fully differentiable.

Getting Started
===============

To get started with this code, clone the repository and install the required dependencies. Then, you can run the main script to see the iLQR in action.

.. code-block:: bash

   git clone git@github.com:ThomasMullen/diffilqrax.git
   cd diffilqrax
   pip install -r requirements.txt
   python -m build
   python setup.py sdist bdist_wheel
   pip install -e .


Structure
=========


Examples
========

.. code-block:: python

   import jax.numpy as jnp
   import jax.random as jr
   from diffilqrax import ilqr
   from diffilqrax.typs import iLQRParams, Theta, ModelDims, System
   from diffilqrax.utils import initialise_stable_dynamics, keygen

   dims = ModelDims(8, 2, 100, dt=0.1)

   key = jr.PRNGKey(seed=234)
   key, skeys = keygen(key, 5)

   Uh = initialise_stable_dynamics(next(skeys), dims.n, dims.horizon, 0.6)[0]
   Wh = jr.normal(next(skeys), (dims.n, dims.m))
   theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(dims.n))
   params = iLQRParams(x0=jr.normal(next(skeys), dims.n), theta=theta)
   Us = jnp.zeros((dims.horizon, dims.m))   
   # define linesearch hyper parameters
   ls_kwargs = {
      "beta":0.8,
      "max_iter_linesearch":16,
      "tol":1e0,
      "alpha_min":0.0001,
      }
   def cost(t, x, u, theta):
      return jnp.sum(x**2) + jnp.sum(u**2)

   def costf(x, theta):
      return jnp.sum(x**2)

   def dynamics(t, x, u, theta):
      return jnp.tanh(theta.Uh @ x + theta.Wh @ u)

   model = System(cost, costf, dynamics, dims)
   ilqr.ilqr_solver(params, model, Us, **ls_kwargs)


License
=======

This project is licensed under the MIT License. See the LICENSE file for details.



Define Lagrangian

.. math::

   \begin{split}
       \mathcal{L}(x,u, \lambda) &= \sum^{T-1}_{t=0} \frac{1}{2} (x_{t}^{T}Q_{t}x_{t} + x_{t}^{T}S_{t}u_{t} + u_{t}^{T}S_{t}^{T}x_{t} + u_{t}^{T}R_{t}u_{t}) + x_{t}^{T}q_{t} + u^{T}_{t}r_{t}  \\ 
       &+ x_{T}^{T}Q_{f}x_{T} + x_{T}^{T}q_{f} \\
       &+ \sum^{T-1}_{t=0} \lambda_{t}^{T}(A_{t}x_{t} + B_{t}u_{t} +a_{t} - \mathbb{I}x_{t+1}) \\
       &+ \lambda_{0}(x_{0} - \mathbb{I}x_{t+1})
   \end{split}

Partial derivatives							

.. math::

   \begin{align}
       \nabla_{x_{t}}\mathcal{L}(x,u, \lambda) &= Q_{t}x_{t} + S_{t}u_{t} + q_{t} + A_{t}^{T}\lambda_{t+1} - \lambda_{t}= 0 \\
       \nabla_{x_{T}} \mathcal{L}(x,u, \lambda)&= Q_{f}x_{T} + q_{f} - \lambda_{T} = 0 \\
       \nabla_{\lambda_{0}}\mathcal{L}(x,u, \lambda) &= x_{0} - \mathbb{I}x_{0} = 0 \\
       \nabla_{\lambda_{t+1}}\mathcal{L}(x,u, \lambda) &= A_{t}x_{t} + B_{t}u_{t} +a_{t}- \mathbb{I}x_{t+1} = 0 \\
       \nabla_{u_{t}}\mathcal{L}(x,u,\lambda) &= S_{t}^{T}x_{t} + R_{t}u_{t} + r_{t}+ B_{t}^{T}\lambda_{t+1} = 0.
   \end{align}
