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


Structure
=========


Examples
========



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
