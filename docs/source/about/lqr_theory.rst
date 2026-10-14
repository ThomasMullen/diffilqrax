LQR Optimization problem
========================

In general, a control optimization problem consists of a cost function, :math:`\mathcal{J}`, 
and a dynamical constraint, :math:`f_k`, that describes how the state evolves over time,

.. math::

   \min \mathcal{J}(x, u) = \ell_f(x_T) + \sum_{t=0}^{T-1} \ell_t(x_t, u_t)

subject to

.. math::

   x_{t+1} = f_t(x_t, u_t), \quad t = 0, \ldots, T-1

   x_0 = x_{\text{init}}

where :math:`\ell_k(\cdot, \cdot)` is the momentary cost function at time :math:`t` 
for a given input :math:`u_t \in \mathbb{R}^m` and state :math:`x_t \in \mathbb{R}^n`. 
The goal is to find an optimal state feedback law :math:`u_t^*(x_t)`, 
that minimizes the cost function :math:`\mathcal{J}` given the control sequence 
:math:`u^* = \{u_0^*, \ldots, u_{T-1}^*\}`.

Some optimal control problems can be solved using the Dynamic Programming (DP) approach.
This method consists of defining a Value function, :math:`\mathcal{V}_t(x, t)`,

.. math::

   \mathcal{V}_t(x, t) = \min_{u_t \in \mathcal{U}} \{ \ell_t(x_t, u_t) + \mathcal{V}_{t+1}(x, t+1) \}

This is the optimal (minimal) cost-to-go from time :math:`t` to the terminal time :math:`T`. 
In order to initialize the Value function, the terminal boundary condition is defined as 
:math:`\mathcal{V}_T(x_T) = \ell_T(x_T)`. Therefore, we can dynamically sweep back through time.

Using the Value function, we can find the optimal control law by finding the argmin of the Value function,

.. math::

   u_t(x_t) = \arg\min_{u_t \in \mathcal{U}} \{ \ell_t(x_t, u_t) + \mathcal{V}_{t+1}(x, t+1) \}

and successively compute the optimal trajectory :math:`\{x_s^*:T\}` in a single forward pass,

.. math::

   x_{t+1}^* = f_t(x_t^*, u_t^*(x_t^*)) = f_t(x_t^*).

Linear quadratic program
------------------------

Linear Quadratic Regulators (LQR) are a particular class of optimal control problems 
that are well-structured and can be solved efficiently using quadratic programming. 
For the LQR problem, the :math:`\ell_k` term in Equation (2.1) is defined as

.. math::

   \ell_t(x_t, u_t) = 
   \begin{bmatrix}
   x_t \\
   u_t
   \end{bmatrix}^T
   \begin{bmatrix}
   Q_t & S_t \\
   S_t^T & R_t
   \end{bmatrix}
   \begin{bmatrix}
   x_t \\
   u_t
   \end{bmatrix}
   +
   \begin{bmatrix}
   x_t \\
   u_t
   \end{bmatrix}^T
   \begin{bmatrix}
   q_t \\
   r_t
   \end{bmatrix} 

and the terminal cost term,

.. math::

   \ell_T(x_T) = x_T^T Q_T x_T + x_T^T q_T.

