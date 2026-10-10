"""
Module solves the differential iterative linear quadratic regulator (DiLQR) problem.
"""

from jax import Array, lax
import jax
import jax.numpy as jnp

from diffilqrax.typs import iLQRParams, System, LQR, LQRParams
from diffilqrax.ilqr import ilqr_solver, approx_lqr_offset
from diffilqrax.diff_lqr import dllqr
from diffilqrax.lqr import bmm
from diffilqrax.utils import quadratise


def add_dynamics_curvature(
    model: System, local_lqr: LQR, Xs: Array, Us: Array, Lambs: Array, params: iLQRParams
) -> LQR:
    """
    Replace the cost Hessians of the local LQR with the Hessians of the Lagrangian.

    Adds the dynamics curvature term λ_{t+1}^T ∇²f(x_t, u_t) to Q, S and R, and shifts the linear
    terms q and r so the LQR's KKT residual at (Xs, Us) and its derivative wrt the parameters are
    unchanged. The resulting LQR has the exact KKT Jacobian of the non-linear problem, so the
    implicit gradients through :func:`diffilqrax.diff_lqr.dllqr` are exact.

    Parameters
    ----------
    model : System
        The system model.
    local_lqr : LQR
        LQR approximation with offsets around (Xs, Us), from :func:`approx_lqr_offset`.
    Xs : Array
        Optimal state trajectory [(T+1)xn].
    Us : Array
        Optimal input trajectory [Txm].
    Lambs : Array
        Optimal adjoint trajectory [(T+1)xn].
    params : iLQRParams
        The iLQR parameters.

    Returns
    -------
    LQR
        Local LQR with Lagrangian Hessians.
    """
    tps = jnp.arange(model.dims.horizon)

    def lamb_dyn(t, x, u, lamb):
        return lamb @ model.dynamics(t, x, u, params.theta)

    (Hxx, Hxu), (_, Huu) = jax.vmap(quadratise(lamb_dyn), in_axes=(0, 0, 0, 0))(
        tps, Xs[:-1], Us, lax.stop_gradient(Lambs[1:])
    )
    return LQR(
        A=local_lqr.A,
        B=local_lqr.B,
        a=local_lqr.a,
        Q=local_lqr.Q + Hxx,
        q=local_lqr.q - bmm(Hxx, Xs[:-1]) - bmm(Hxu, Us),
        R=local_lqr.R + Huu,
        r=local_lqr.r - bmm(Huu, Us) - bmm(Hxu.transpose(0, 2, 1), Xs[:-1]),
        S=local_lqr.S + Hxu,
        Qf=local_lqr.Qf,
        qf=local_lqr.qf,
    )()


def dilqr(model: System, params: iLQRParams, Us_init: Array, exact_hessian: bool = True, **kwargs) -> Array:
    """
    Solves the differential iLQR problem.

    Parameters
    ----------
    model : System
        The system model.
    params : iLQRParams
        The iLQR parameters.
    Us_init : Array
        The initial control sequence.
    exact_hessian : bool, optional
        If True (default), the implicit gradients use the Hessian of the Lagrangian, including
        the dynamics curvature λ^T ∇²f, and match direct differentiation. If False, only the cost
        Hessians are used, as in the OCaml ``dilqr`` implementation; this is exact only for
        linear dynamics.
    **kwargs
        Additional keyword arguments for the iLQR solver.

    Returns
    -------
    Array
        The optimized state-control trajectory [(T+1)x(n+m)].
    """
    sol = ilqr_solver(model, lax.stop_gradient(params), Us_init, **kwargs)
    (Xs_star, Us_star, Lambs_star), _, _ = sol
    tau_star = jnp.c_[Xs_star[:, ...], jnp.r_[Us_star, jnp.zeros(shape=(1, model.dims.m))]]
    local_lqr = approx_lqr_offset(model, Xs_star, Us_star, params)  ##equiv of g1
    if exact_hessian:
        local_lqr = add_dynamics_curvature(model, local_lqr, Xs_star, Us_star, Lambs_star, params)
    params = LQRParams(lqr=local_lqr, x0=params.x0)
    tau_star = dllqr(model.dims, params, tau_star)
    return tau_star
