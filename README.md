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