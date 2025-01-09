![DiffiLQRax logo](./doc/source/_static/images/diffilqrax_logo_banner_dm.png)


![Pylint](https://github.com/ThomasMullen/diffilqrax/actions/workflows/pylint.yml/badge.svg)
![Python Package](https://github.com/ThomasMullen/diffilqrax/actions/workflows/python-package.yml/badge.svg)
![Python Publish](https://github.com/ThomasMullen/diffilqrax/actions/workflows/python-publish.yml/badge.svg)

# Diffilqrax: Differentiable optimal control

## Diffilqrax: What is it?

This repository contains an implementation of the iterative Linear Quadratic Regulator (iLQR) using the JAX library. The iLQR is a powerful algorithm used for optimal control, and this implementation is designed to be fully differentiable.

## Getting Started

To get started with this code, clone the repository and install the required dependencies. Then, you can run the main script to see the iLQR in action.

```bash
git clone git@github.com:ThomasMullen/diffilqrax.git
cd diffilqrax
python -m build
pip install -e .
```

or, you can import from pip install

```bash
pip install diffilqrax
```

### Quick example

```python
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
theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros(dims.n), Q=jnp.eye(dims.n))
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
```

## Design principle

![DiffiLQRax Deisgn Principle](./doc/source/_static/images/diffilqrax_design_principle.png)


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement #813457.