# ilqr_vae_py
jax ilqr implementation


Define Lagrangian
$$
\begin{split}
    \mathcal{L}(\bm{x},\bm{u}, \bm{\lambda}) &= \sum^{T-1}_{t=0} \frac{1}{2} (x_{t}^{T}Q_{t}x_{t} + x_{t}^{T}S_{t}u_{t} + u_{t}^{T}S_{t}^{T}x_{t} + u_{t}^{T}R_{t}u_{t}) + x_{t}^{T}q_{t} + u^{T}_{t}r_{t}  \\ 
    &+ x_{T}^{T}Q_{f}x_{T} + x_{T}^{T}q_{f} \\
    &+ \sum^{T-1}_{t=0} \lambda_{t}^{T}(A_{t}x_{t} + B_{t}u_{t} +a_{t} - \mathbb{I}x_{t+1}) \\
    &+ \lambda_{0}(x_{0} - \mathbb{I}x_{t+1})
\end{split}
$$

Partials
$$
\begin{align}
	\nabla_{x_{t}}\mathcal{L}(x,u, \bm{\lambda}) &= Q_{t}x_{t} + S_{t}u_{t} + q_{t} + A_{t}^{T}\bm{\lambda}_{t+1} - \bm{\lambda}_{t}= 0 \\
	\nabla_{x_{T}} \mathcal{L}(x,u, \bm{\lambda})&= Q_{f}x_{T} + q_{f} - \bm{\lambda}_{T} = 0 \\
	\nabla_{\bm{\lambda}_{0}}\mathcal{L}(x,u, \bm{\lambda}) &= x_{0} - \mathbb{I}x_{0} = 0 \\
	\nabla_{\bm{\lambda}_{t+1}}\mathcal{L}(x,u, \bm{\lambda}) &= A_{t}x_{t} + B_{t}u_{t} +a_{t}- \mathbb{I}x_{t+1} = 0 \\
	\nabla_{u_{t}}\mathcal{L}(x,u,\bm{\lambda}) &= S_{t}^{T}x_{t} + R_{t}u_{t} + r_{t}+ B_{t}^{T}\lambda_{t+1} = 0.
\end{align}
$$