
Associative iLQR
==================


Implementation of the Parallel Linear Quadratic Regulator (PLQR) algorithm
--------------------------------------------------------------------------
1. Initialisation: compute elements :math:`a=\{A, b, C, η, J\}`
   do for all in parallel i.e. :code:`vmap`;
2. Parallel backward scan: initialise with all elements & apply associative operator
   note association operator should be vmap. Scan will return :math:`V_{k}(x_{k})=\{V, v\}`;
3. Compute optimal control: :math:`u_k = -K_kx_k + K^{v}_{k} v_{k+1} - K_k^{c} c_{k}`.
   :math:`K`s have closed form solutions, so calculate :math:`u_k` in parallel :code:`vmap`.

.. automodule:: diffilqrax.parallel_ilqr
   :members:
   :undoc-members:
   :show-inheritance: