# ilqr_vae_py
jax ilqr implementation

LQR linear system with quadratic and linear cost terms

Final state cost:
$$l_T(x_T) = \frac{1}{2} x_T^{T}Q_Tx_T + q_T^{T}x_T+\alpha_{T}$$

Current state cost:
$$l_t(x_t) = \frac{1}{2} x_t^{T}Q_tx_t + \frac{1}{2} u_t^{T}R_tu_t + q_t^{T}x_t + r_t^{T}u_t+\alpha_{t}$$

With dynamics:
$$f_t(x_t, u_t) = A_tx_t + B_tu_t + a_t$$

The optimal cost-to-go $J^{*}_T{}(x_t)$ is defined as:

$$J^{*}_T{}(x_t) = \frac{1}{2} x_T^{T}P_Tx_T + p_T^{T}x_T+\beta_{T} $$

where $P$ and $p$ are the quadratic and linear value iteration respectively.




# Resources
DDP implementation https://bjack205.github.io/papers/AL_iLQR_Tutorial.pdf
Jax DP and implicit differentiation https://stephentu.github.io/presentations/jax4dc/#/1



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