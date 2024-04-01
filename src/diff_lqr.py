from functools import partial
import jax.lax as lax
from jax.lax import batch_matmul as bmm
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Tuple
from jax import Array, custom_vjp
from .lqr import (
    LQR,
    Params,
    ModelDims,
    solve_lqr, symmetrise_tensor
)


def get_qra_bar(params: Params, dims: ModelDims,tau_bar: Array) -> Tuple[Array, Array, Array]:
    """Helper function to get gradients wrt to q, r, a."""
    # q_bar, r_bar, a_bar from solving the rev LQR problem where q_rev = x_bar, r_rev = u_bar, a_rev = lambda_bar (set to 0 here)
    lqr = params.lqr
    n = dims.n
    x_bar, u_bar = tau_bar[..., :n], tau_bar[..., n:]
    # Lambs_bar = jnp.zeros_like(lqr.a)
    # set-up LQR problem with r_rev = u_bar, q_rev = x_bar, a_rev = lambda_bar (set to 0 here)
    swapped_lqr = LQR(A=lqr.A, B=lqr.B, a=jnp.zeros_like(lqr.a), 
                      Q=lqr.Q, q=x_bar, 
                      Qf=lqr.Qf, qf=lqr.qf, 
                      R=lqr.R, r=u_bar, S=lqr.S)
    
    swapped_params = Params(params.x0, swapped_lqr)
    _, q_bar, r_bar, a_bar = solve_lqr(swapped_params, dims)
    ##TODO : check the indices for a_bar
    return q_bar, r_bar, a_bar


@partial(custom_vjp, nondiff_argnums=(0,1,))
def dlqr(params: Params, dims: ModelDims) -> Tuple[Array, Array, Array, Array]:
    """params vector contains all the LQR parameters : here, this assumes an LQR problem
    In the more general case, depending on where we are defining this, we may to take into account the fact that we are at around the solution, so the effective problem has extra linear terms, as follows :

    local_LQR = params.lqr
    local_LQR.q = local_LQR.q - bmm(Xs_star, local_LQR.Q) - bmm(Us_star, local_LQR.S)
    local_LQR.r = local_LQR.r - bmm(Xs_star, np.transpose(local_LQR.S, axis = (0,2,1))) - bmm(Us_star, local_LQR.R)
    params.lqr = local_LQR"""
    return solve_lqr(params, dims)


def fwd_dlqr(params: Params, dims: ModelDims):
    _, Xs_star, Us_star, Lambs = dlqr(params, dims)
    tau_star = jnp.concatenate([Xs_star, jnp.concatenate([Us_star, jnp.zeros_like(Us_star)[0]], axis = 0)], axis=1)
    return tau_star, (params, dims, Lambs, tau_star)


def rev_dlqr(res, tau_bar: Array) -> Params:
    params, dims, Lambs, tau_star = res
    """
  Inputs : params (contains lqr parameters, x0), tau_star_bar (gradients wrt to tau at tau_star)
  params : LQR(A, B, a, Q, q, Qf, qf, R, r, S)
  A : T x N x N
  B : T x N x M
  a : T x N x 1
  q : T x N x 1
  Q : T x N x N
  R : T x M x M
  r : T x M x 1
  S : T x N x M
  
  - q_bar, r_bar, a_bar from solving the rev LQR problem where q_rev = x_bar, r_rev = u_bar, a_rev = lambda_bar (set to 0 here)
  - define c_bar = [q_bar, r_bar]
  - define F_bar (where F = [A, B] and C_bar (where C = [Q,R]) as C_bar 0.5*(c_bar tau_star.T + tau_star c_bar.T))
  - F_bar_t = lambda_star_{t+1}c_bar.T + f_{t+1} tau_star_t.T
  
  Returns : params_bar, i.e tuple with gradients wrt to x0, LQR params, and horizon
  #TODO :
  - make sure we are solving the local problenn, with the local parameters 
  - we don't want to differentiate wrt to the horizon so maybe we shouldn't use the vector containing it
  """
    # this asssumes we are passing dimension parameters
    n = dims.n # LQR.A[0].shape()[1], LQR.B[0].shape()[1]
    q_bar, r_bar, a_bar = get_qra_bar(params, dims, tau_bar)
    c_bar = jnp.concatenate([q_bar, r_bar], axis=1)
    F_bar = bmm(Lambs[1:], jnp.transpose(c_bar[:-1], axis=(0, 2, 1))) + bmm(
        a_bar[1:], jnp.transpose(tau_star[:-1], axis=(0, 2, 1))
    )
    C_bar = 0.5 * (symmetrise_tensor(bmm(c_bar, tau_star.T))) 
    Q_bar, R_bar = C_bar[:, :n, :n], C_bar[:, n:, n:]
    S_bar = 0.5 * (symmetrise_tensor(C_bar[:, :n, n:])) 
    A_bar, B_bar = F_bar[..., :n], F_bar[..., n:]
    LQR_bar = LQR(
        A=A_bar,
        B=B_bar,
        a=a_bar,
        Q=Q_bar[:-1],
        q=q_bar[:-1],
        Qf=Q_bar[-1][None, ...],
        qf=q_bar[-1][None, ...],
        R=R_bar,
        r=r_bar,
        S=S_bar,
    )
    x0_bar = jnp.zeros_like(params.x0)
    return Params(x0=x0_bar, lqr=LQR_bar)


dlqr.defvjp(fwd_dlqr, rev_dlqr)


"""Useful pieces of code : 


@partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point(f, a, x_guess):
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.abs(x_prev - x) > 1e-6

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess)))
  return x_star

def fixed_point_fwd(f, a, x_init):
  x_star = fixed_point(f, a, x_init)
  return x_star, (a, x_star)

def fixed_point_rev(f, res, x_star_bar):
  a, x_star = res
  _, vjp_a = vjp(lambda a: f(a, x_star), a)
  a_bar, = vjp_a(fixed_point(partial(rev_iter, f),
                             (a, x_star, x_star_bar),
                             x_star_bar))
  return a_bar, jnp.zeros_like(x_star)
  
def rev_iter(f, packed, u):
  a, x_star, x_star_bar = packed
  _, vjp_x = vjp(lambda x: f(a, x), x_star)
  return x_star_bar + vjp_x(u)[0]

fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)



 let g1 ~theta =
    let ffb = ffb ~theta in
    fun ~x0 ~ustars ->
      let flxx, flx, tape, xf = ffb x0 ustars in
      let lambda0, lambdas = Lqr.adjoint flx tape in
      let lambdas = AD.Maths.stack ~axis:0 (Array.of_list (lambda0 :: lambdas)) in
      let big_taus = [ AD.Maths.concatenate ~axis:1 [| xf; AD.Mat.zeros 1 m |] ] in
      let big_fs = [ AD.Mat.zeros (n + m) n ] in
      let big_cs =
        let row1 = AD.Maths.(concatenate ~axis:1 [| flxx; AD.Mat.zeros n m |]) in
        let row2 = AD.Mat.zeros m (n + m) in
        [ AD.Maths.concatenate ~axis:0 [| row1; row2 |] ]
      in
      let cs =
        [ AD.Maths.concatenate
            ~axis:1
            [| AD.Maths.(flx - (xf *@ flxx)); AD.Mat.zeros 1 m |]
        ]
      in
      let fs = [] in
      let big_taus, big_fs, big_cs, cs, fs, _ =
        List.fold_left
          (fun (taus, big_fs, big_cs, cs, fs, next_x) (s : Lqr.t) ->
            ignore next_x;
            let taus =
              let tau = AD.Maths.concatenate ~axis:1 [| s.x; s.u |] in
              tau :: taus
            in
            let big_f = AD.Maths.(concatenate ~axis:0 [| s.a; s.b |]) in
            let big_c =
              let row1 = AD.Maths.(concatenate ~axis:1 [| s.rlxx; transpose s.rlux |]) in
              let row2 = AD.Maths.(concatenate ~axis:1 [| s.rlux; s.rluu |]) in
              AD.Maths.(concatenate ~axis:0 [| row1; row2 |])
            in
            let c =
              AD.Maths.(
                concatenate
                  ~axis:1
                  [| s.rlx - (s.x *@ s.rlxx) - (s.u *@ s.rlux)
                   ; s.rlu - (s.u *@ s.rluu) - (s.x *@ transpose s.rlux)
                  |])
            in
            taus, big_f :: big_fs, big_c :: big_cs, c :: cs, s.f :: fs, s.x)
          (big_taus, big_fs, big_cs, cs, fs, xf)
          tape
      in
      let taus = AD.Maths.stack ~axis:0 Array.(of_list big_taus) in
      let big_fs = AD.Maths.stack ~axis:0 Array.(of_list big_fs) in
      let big_cs = AD.Maths.stack ~axis:0 Array.(of_list big_cs) in
      let cs = AD.Maths.stack ~axis:0 Array.(of_list cs) in
      let fs = AD.Maths.stack ~axis:0 Array.(of_list fs) in
      taus, big_fs, big_cs, cs, lambdas, fs
      
      
let g2 =
    let swap_out_tape tape tau_bar =
      (* swapping out the tape *)
      let _, tape =
        List.fold_left
          (fun (k, tape) (s : Lqr.t) ->
            let rlx =
              AD.Maths.(
                reshape
                  (AD.Maths.get_slice [ [ k ]; []; [ 0; pred n ] ] tau_bar)
                  [| 1; n |])
            in
            let rlu =
              AD.Maths.reshape
                (AD.Maths.get_slice [ [ k ]; []; [ n; -1 ] ] tau_bar)
                [| 1; m |]
            in
            succ k, Lqr.{ s with rlu; rlx } :: tape)
          (0, [])
          (* the tape is backward in time hence we reverse it *)
          (List.rev tape)
      in
      let flx =
        AD.Maths.reshape
          (AD.Maths.get_slice [ [ -1 ]; []; [ 0; n - 1 ] ] tau_bar)
          [| 1; n |]
      in
      flx, tape
    in
    fun ~theta ->
      let ffb = ffb ~theta in
      fun ~taus ~ustars ~lambdas ->
        let ds ~x0 ~tau_bar =
          (* recreating tape, pass as argument in the future *)
          let flxx, _, tape, _ = ffb x0 ustars in
          let flx, tape = swap_out_tape tape tau_bar in
          let acc, _ = Lqr.backward flxx flx tape in
          let ctbars_xf, ctbars_tape = Lqr.forward acc AD.Mat.(zeros 1 n) in
          let dlambda0, dlambdas = Lqr.adjoint_back ctbars_xf flxx flx ctbars_tape in
          let ctbars =
            List.map
              (fun (s : Lqr.t) -> AD.Maths.(concatenate ~axis:1 [| s.x; s.u |]))
              ctbars_tape
            |> List.cons AD.Maths.(concatenate ~axis:1 [| ctbars_xf; AD.Mat.zeros 1 m |])
            |> List.rev
          in
          ( AD.Maths.stack ~axis:0 (Array.of_list ctbars)
          , AD.Maths.stack ~axis:0 (Array.of_list (dlambda0 :: dlambdas)) )
        in
        let big_ft_bar ~taus ~lambdas ~dlambdas ~ctbars () =
          let tdl =
            Bmo.AD.bmm
              (AD.Maths.transpose
                 ~axis:[| 0; 2; 1 |]
                 (AD.Maths.get_slice [ [ 0; -2 ]; []; [] ] taus))
              (AD.Maths.get_slice [ [ 1; -1 ]; []; [] ] dlambdas)
          in
          let dtl =
            Bmo.AD.bmm
              (AD.Maths.transpose
                 ~axis:[| 0; 2; 1 |]
                 AD.Maths.(get_slice [ [ 0; -2 ]; []; [] ] ctbars))
              (AD.Maths.get_slice [ [ 1; -1 ]; []; [] ] lambdas)
          in
          let output = AD.Maths.(tdl + dtl) in
          AD.Maths.concatenate ~axis:0 [| output; AD.Arr.zeros [| 1; n + m; n |] |]
        in
        let big_ct_bar ~taus ~ctbars () =
          let tdt = Bmo.AD.bmm (AD.Maths.transpose ~axis:[| 0; 2; 1 |] ctbars) taus in
          AD.Maths.(F 0.5 * (tdt + transpose ~axis:[| 0; 2; 1 |] tdt))
        in
        build_aiso
          (module struct
            let label = "g2"
            let ff _ = AD.primal' taus
            let df _ _ _ _ = raise (Owl_exception.NOT_IMPLEMENTED "g2 forward mode")

            let dr idxs x _ ybar =
              let x0 = x.(4) in
              let ctbars, dlambdas = ds ~x0 ~tau_bar:!ybar in
              List.map
                (fun idx ->
                  if idx = 0
                  then big_ft_bar ~taus ~lambdas ~dlambdas ~ctbars ()
                  else if idx = 1
                  then big_ct_bar ~taus ~ctbars ()
                  else if idx = 2
                  then ctbars
                  else if idx = 3
                  then AD.Maths.(get_slice [ [ 1; -1 ] ] dlambdas)
                  else
                    AD.Maths.(get_slice [ [ 0 ] ] dlambdas)
                    |> fun x -> AD.Maths.reshape x [| 1; -1 |])
                idxs
          end : Aiso)
"""
