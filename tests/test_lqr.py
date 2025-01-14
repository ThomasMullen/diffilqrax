"""
Unit test for the LQR module
"""

from pathlib import Path
import unittest
from os import getcwd
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from matplotlib.pyplot import subplots, close, style

from diffilqrax.typs import (
    LQR,
    LQRParams,
    ModelDims,
)
from diffilqrax.lqr import (
    simulate_trajectory,
    lqr_adjoint_pass,
    lin_dyn_step,
    lqr_forward_pass,
    lqr_backward_pass,
    solve_lqr,
    kkt,
)
from diffilqrax.exact import quad_solve, exact_solve
from diffilqrax.utils import keygen, initialise_stable_dynamics

jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)  # double precision

OUTPUT_FIGS=False # save figures of tests
if OUTPUT_FIGS:
    FIG_DIR = Path(Path(getcwd()), "fig_dump", "seq_lqr")
    FIG_DIR.mkdir(exist_ok=True,parent=True)
    PLOT_URL = ("https://gist.githubusercontent.com/"
        "ThomasMullen/e4a6a0abd54ba430adc4ffb8b8675520/"
        "raw/1189fbee1d3335284ec5cd7b5d071c3da49ad0f4/"
        "figure_style.mplstyle")
    style.use(PLOT_URL)

def _plot_trajs(*args):
    x, u, lamb = args
    fig, ax = subplots(1,3,figsize=(10,3))
    ax[0].plot(x)
    ax[0].set(ylabel="x")
    ax[1].plot(u)
    ax[1].set(ylabel="u")
    ax[2].plot(lamb)
    ax[2].set(ylabel="lamb")
    fig.tight_layout()
    return fig

def _plot_kkt(*args):
    """plot KKT conditions and state trajectories"""
    x, u, lamb, dl_dx, dl_du, dl_dlamb = args
    fig, ax = subplots(2,3, figsize=(10,3), sharey=False)
    ax[0,0].plot(x.squeeze())
    ax[0,0].set(title="X")
    ax[0,1].plot(u.squeeze())
    ax[0,1].set(title="U")
    ax[0,2].plot(lamb.squeeze())
    ax[0,2].set(title="λ")
    ax[1,0].plot(dl_dx.squeeze())
    ax[1,0].set(title="dLdX")
    ax[1,1].plot(dl_du.squeeze())
    ax[1,1].set(title="dLdUs")
    ax[1,2].plot(dl_dlamb.squeeze())
    ax[1,2].set(title="dLdλ")
    fig.tight_layout()
    return fig

def _plot_lqr_sols(*args):
    """plot optimal state trajectories for diff sols"""
    x_dir, u_dir, x_cg, u_cg, x_inv, u_inv = args
    fig, ax = subplots(2,3, figsize=(10,3), sharey=False)
    ax[0,0].set(title="LQR solve", ylabel="X")
    ax[0,0].plot(x_dir.squeeze())
    ax[1,0].set(ylabel="U")
    ax[1,0].plot(u_dir.squeeze())
    ax[0,1].set(title="CG solve", ylabel="X")
    ax[0,1].plot(x_cg.squeeze())
    ax[1,1].set(ylabel="U")
    ax[1,1].plot(u_cg.squeeze())
    ax[0,2].set(title="Inv solve",ylabel="X")
    ax[0,2].plot(x_inv.squeeze())
    ax[1,2].set(ylabel="U")
    ax[1,2].plot(u_inv.squeeze())
    fig.tight_layout()
    return fig

def is_jax_array(arr: Array)->bool:
    """validate jax array type"""
    return isinstance(arr, jnp.ndarray) and not isinstance(arr, onp.ndarray)


def setup_lqr(dims: chex.Dimensions,
              pen_weight: dict = {"Q": 1e-0, "R": 1e-3, "Qf": 1e0, "S": 1e-3}) -> LQR:
    """Setup LQR problem"""
    key = jr.PRNGKey(seed=234)
    key, skeys = keygen(key, 3)
    # initialise dynamics
    span_time_m=dims["TXX"]
    span_time_v=dims["TX"]
    A = initialise_stable_dynamics(next(skeys), *dims['NT'],radii=0.1)
    B = jnp.tile(jr.normal(next(skeys), dims['NM']), span_time_m)
    a = jnp.tile(jr.normal(next(skeys), dims['N']), span_time_v)
    # define cost matrices
    Q = pen_weight["Q"] * jnp.tile(jnp.eye(dims['N'][0]), span_time_m)
    q = 1e-1 * jnp.asarray([jnp.cos(t/5)*jnp.arange(dims["N"][0]) for t in range(dims["T"][0])])
    R = pen_weight["R"] * jnp.tile(jnp.eye(dims['M'][0]), span_time_m)
    r = 1e-6 * jnp.tile(jnp.ones(dims['M']), span_time_v)
    S = pen_weight["S"] * jnp.tile(jnp.ones(dims['NM']), span_time_m)
    Qf = pen_weight["Q"] * jnp.eye(dims['N'][0])
    qf = 2*1e-1 * jnp.ones(dims['N'])
    # construct LQR
    lqr = LQR(A, B, a, Q, q, R, r, S, Qf, qf)
    return lqr()


class TestLQR(unittest.TestCase):
    """Test LQR dimensions and dtypes"""

    def setUp(self):
        """Instantiate dummy LQR"""
        print("\nRunning setUp method...")
        n = 30
        self.dims = chex.Dimensions(T=60, N=n, M=15, X=1)
        self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.1)
        print("Model dimensionality", self.dims["TNMX"])
        print("\nMake LQR struct")
        self.lqr = setup_lqr(self.dims)

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.ones((n,))
        Us = jnp.zeros(self.dims["TM"], dtype=float)
        Us = Us.at[2].set(1.0)
        self.Us = Us

    def test_lqr_params_struct(self):
        """test LQRParams struct instances"""
        print("Construct params")
        params = LQRParams(self.x0, self.lqr)
        chex.assert_trees_all_close(params[0], self.x0)
        chex.assert_trees_all_close(params.x0, self.x0)
        chex.assert_type(params.x0, float)
        assert isinstance(params.lqr, LQR)
        assert isinstance(params[1], LQR)
        assert isinstance(params[-1], LQR)
        print("LQRParams struct passed.")

    def test_lqr_struct(self):
        """Test test shape of LQR"""
        print("Running test_lqr_struct")

        # test cost is positive symmetric
        chex.assert_trees_all_close(self.lqr.Q[0], self.lqr.Q[0].T)
        chex.assert_trees_all_close(self.lqr.R[0], self.lqr.R[0].T)
        # check if Q is positive definite
        assert jnp.all(jnp.linalg.eigvals(self.lqr.Q[0]) > 0), "Q is not positive definite"
        assert jnp.all(jnp.linalg.eigvals(self.lqr.R[0]) > 0), "R is not positive definite"

        # check shape
        chex.assert_shape(self.lqr.A, self.dims["TNN"])
        chex.assert_shape(self.lqr.B, self.dims["TNM"])
        chex.assert_shape(self.lqr.Q, self.dims["TNN"])
        chex.assert_shape(self.lqr.R, self.dims["TMM"])
        chex.assert_shape(self.lqr.S, self.dims["TNM"])

        # check dtypes
        chex.assert_type(self.lqr.S.dtype, float)
        chex.assert_type(self.lqr.Qf.dtype, float)
        # test jax arrays
        assert is_jax_array(self.lqr.Q)
        assert is_jax_array(self.lqr.R)
        assert is_jax_array(self.lqr.S)
        assert is_jax_array(self.lqr.Qf)
        assert is_jax_array(self.lqr.A)
        assert is_jax_array(self.lqr.B)
        print("LQR struct passed.")

    def test_simulate_trajectory(self):
        """test simulate trajectory shape and dtype"""
        print("Running test_simulate_trajectory")
        params = LQRParams(self.x0, self.lqr)
        Xs = simulate_trajectory(lin_dyn_step, self.Us, params, self.sys_dims)
        chex.assert_type(Xs, float)
        chex.assert_shape(Xs, (self.dims["T"][0] + 1,) + self.dims["N"])

    def test_lqr_adjoint_pass(self):
        """test adjoint pass shape and dtype"""
        print("Running test_lqr_adjoint_pass")
        params = LQRParams(self.x0, self.lqr)
        Xs_sim = simulate_trajectory(lin_dyn_step, self.Us, params, self.sys_dims)
        Lambs = lqr_adjoint_pass(Xs_sim, self.Us, params)
        chex.assert_type(Lambs, float)
        chex.assert_shape(Lambs, (self.dims["T"][0] + 1,) + self.dims["N"])

    def test_lqr_backward_pass(self):
        """test backward pass shape and dtype"""
        params = LQRParams(self.x0, self.lqr)
        dJ, Ks = lqr_backward_pass(lqr=params.lqr)
        chex.assert_type(Ks.K, float)
        chex.assert_shape(Ks.K, self.dims["TMN"])
        chex.assert_type(Ks.k, float)
        chex.assert_shape(Ks.k, self.dims["TM"])
        chex.assert_type(dJ, float)

    def test_lqr_forward_pass(self):
        """test forward pass shape and dtype"""
        params = LQRParams(self.x0, self.lqr)
        dJ, Ks= lqr_backward_pass(lqr=params.lqr)
        Xs_lqr, Us_lqr = lqr_forward_pass(gains=Ks, params=params)
        chex.assert_type(Xs_lqr, float)
        chex.assert_shape(Xs_lqr, (self.dims["T"][0] + 1,) + self.dims["N"])
        chex.assert_type(Us_lqr, float)
        chex.assert_shape(Us_lqr, self.dims["TM"])
        chex.assert_type(Ks.K, float)
        chex.assert_shape(Ks.K, self.dims["TMN"])
        chex.assert_type(Ks.k, float)
        chex.assert_shape(Ks.k, self.dims["TM"])
        assert is_jax_array(Ks.K)
        assert is_jax_array(Ks.k)

    def test_solve_lqr(self):
        """test LQR solution shape and dtype"""
        params = LQRParams(self.x0, self.lqr)
        Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params)
        chex.assert_type(Xs_lqr, float)
        chex.assert_shape(Xs_lqr, (self.dims["T"][0] + 1,) + self.dims["N"])
        chex.assert_type(Us_lqr, float)
        chex.assert_shape(Us_lqr, self.dims["TM"])
        chex.assert_type(Lambs_lqr, float)
        chex.assert_shape(Lambs_lqr, (self.dims["T"][0] + 1,) + self.dims["N"])
        # verify output jax arrays
        assert is_jax_array(Xs_lqr)
        assert is_jax_array(Us_lqr)
        assert is_jax_array(Lambs_lqr)
    
    def test_solution_output(self):
        """test LQR solution output"""
        params = LQRParams(self.x0, self.lqr)
        Xs_lqr, Us_lqr, Lambs_lqr = solve_lqr(params)
        if OUTPUT_FIGS:
            fig = _plot_trajs(Xs_lqr, Us_lqr, Lambs_lqr)
            fig.savefig(f"{FIG_DIR}/seq_lqr_sol{self.sys_dims.n:03}_{self.sys_dims.m:03}.png")
            close()

    def test_kkt_optimal(self):
        """test KKT conditions for LQR optimality"""
        # Setup the LQR problem
        params = LQRParams(self.x0, self.lqr)
        Xs_dir, Us_dir, Lambs_dir = solve_lqr(params=params)
        # Exercise the KKT function
        dLdXs, dLdUs, dLdLambs = kkt(params, Xs_dir, Us_dir, Lambs_dir)
        if OUTPUT_FIGS:
            # Plot the KKT residuals
            fig = _plot_kkt(Xs_dir, Us_dir, Lambs_dir, dLdXs, dLdUs, dLdLambs)
            fig.savefig(f"{FIG_DIR}/seq_lqr_kkt_{self.sys_dims.n:03}_{self.sys_dims.m:03}.png")
            close()
            
        # Verify that the average KKT conditions are satisfied
        assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-05, atol=1e-08)

        # Verify that the terminal state KKT conditions is satisfied
        assert jnp.allclose(dLdXs[-1], 0.0,
                            rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"

        # Verify that all KKT conditions are satisfied
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)

    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


class TestLQRSolutionExact(unittest.TestCase):
    """
    Test LQR solution comparing to the exact solution using a CG solve (in a case in which the 
    dynamics are constant)
    """

    def setUp(self):
        """Instantiate LQR example using the pendulum example to compare against Ocaml"""
        print("\nRunning setUp method...")
        self.dims = chex.Dimensions(T=100, N=2, M=2, X=1)
        self.sys_dims = ModelDims(*self.dims["NMT"], dt=0.1)
        dt = self.sys_dims.dt
        self.x_targ = jnp.sin(jnp.linspace(0,2,self.dims["T"][0]+1))

        A = jnp.tile(jnp.array([[1,dt],[-1*dt,1-0.5*dt]]), self.dims["TXX"])
        B = jnp.tile(jnp.array([[0,0],[1,0]]), self.dims["TXX"])*dt
        a = jnp.zeros(self.dims["TN"], dtype=float)
        Q = 2. * jnp.tile(jnp.eye(self.dims["N"][0]), self.dims["TXX"])
        q = 0. * jnp.tile(jnp.ones(self.dims["N"]), self.dims["TX"])
        R = 0.5 * jnp.tile(jnp.eye(self.dims["M"][0]), self.dims["TXX"])
        r = 0. * jnp.tile(jnp.ones(self.dims["M"]), self.dims["TX"])
        S = 0. * jnp.tile(jnp.ones(self.dims["NM"]), self.dims["TXX"])
        Qf = 0. * jnp.eye(self.dims["N"][0])
        qf = 0.   * jnp.ones(self.dims["N"])
        
        self.lqr = LQR(A, B, a, Q, q, R, r, S, Qf, qf)()

        print("\nMake initial state x0 and input U")
        self.x0 = jnp.array([0.3, 0.])
        Us = jnp.zeros(self.dims["TM"]) * 1.0
        self.Us = Us.at[2].set(1.0)
        self.params = LQRParams(self.x0, self.lqr)

    def test_lqr_solution(self):
        """test LQR solution using jaxopt conjugate gradient solution"""
        # setup
        params = LQRParams(self.x0, self.lqr)
        print("Make tmp dir")
        # Exercise the LQR solver function
        Xs_dir, Us_dir, _ = solve_lqr(self.params)
        print("Lqr solve")
        Xs_quad, Us_quad = quad_solve(self.params, self.sys_dims, self.x0)
        print("CG solve")
        Xs_exact, Us_exact = exact_solve(self.params, self.sys_dims, self.x0)
        print("Mat inversion solve")
        if OUTPUT_FIGS:
            print("Plot u solutions")
            fig = _plot_lqr_sols(Xs_dir, Us_dir, Xs_quad, Us_quad, Xs_exact, Us_exact)
            fig.savefig(f"{FIG_DIR}/compare_tiv_lqr_solver.png")
            close()
        # Verify that the two solutions are close
        assert jnp.allclose(Us_dir[:-1], Us_exact, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(Xs_dir[:-1], Xs_exact, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(Xs_dir[:-1], Xs_quad, rtol=1e-04, atol=1e-05)
        assert jnp.allclose(Us_dir[:-1], Us_quad, rtol=1e-04, atol=1e-05)

    def test_kkt_optimal(self):
        """test KKT conditions for LQR optimality"""
        # Setup the LQR problem
        Xs_dir, Us_dir, Lambs_dir = solve_lqr(self.params)
        # Exercise the KKT function
        dLdXs, dLdUs, dLdLambs = kkt(self.params, Xs_dir, Us_dir, Lambs_dir)
        # Plot the KKT residuals
        if OUTPUT_FIGS:
            fig = _plot_kkt(Xs_dir, Us_dir, Lambs_dir, dLdXs, dLdUs, dLdLambs)
            fig.savefig(f"{FIG_DIR}/seq_tiv_lqr_kkt.png")
            close()
            
        # Verify that the average KKT conditions are satisfied
        assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-05, atol=1e-08)

        # Verify that the terminal state KKT conditions is satisfied
        assert jnp.allclose(dLdXs[-1], 0.0,
                            rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"

        # Verify that all KKT conditions are satisfied
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)

    def test_kkt_optimal_bias(self):
        """test KKT conditions for LQR optimality"""
        # Setup the LQR problem
        lqr = self.lqr._replace(a=jnp.ones(self.dims["TN"]))
        Xs_dir, Us_dir, Lambs_dir = solve_lqr(LQRParams(self.x0, lqr))
        # Exercise the KKT function
        dLdXs, dLdUs, dLdLambs = kkt(LQRParams(self.x0, lqr), Xs_dir, Us_dir, Lambs_dir)
        # Plot the KKT residuals
        if OUTPUT_FIGS:
            fig = _plot_kkt(Xs_dir, Us_dir, Lambs_dir, dLdXs, dLdUs, dLdLambs)
            fig.savefig(f"{FIG_DIR}/seq_tivb_lqr_kkt.png")
            close()
            
        # Verify that the average KKT conditions are satisfied
        assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-05, atol=1e-08)

        # Verify that the terminal state KKT conditions is satisfied
        assert jnp.allclose(dLdXs[-1], 0.0,
                            rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"

        # Verify that all KKT conditions are satisfied
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)

    def test_kkt_optimal_tvq(self):
        """test KKT conditions for LQR optimality"""
        # Setup the LQR problem
        lqr = self.lqr._replace(a=jnp.ones(self.dims["TN"]),
                                q=jnp.tile(self.x_targ[:-1], (self.dims["N"][0], 1)).T,
                                qf=jnp.tile(self.x_targ[-1], (self.dims["N"][0])),
                                )
        Xs_dir, Us_dir, Lambs_dir = solve_lqr(LQRParams(self.x0, lqr))
        # Exercise the KKT function
        dLdXs, dLdUs, dLdLambs = kkt(LQRParams(self.x0, lqr), Xs_dir, Us_dir, Lambs_dir)
        # Plot the KKT residuals
        if OUTPUT_FIGS:
            fig = _plot_kkt(Xs_dir, Us_dir, Lambs_dir, dLdXs, dLdUs, dLdLambs)
            fig.savefig(f"{FIG_DIR}/seq_tvq_lqr_kkt.png")
            close()            
        # Verify that the average KKT conditions are satisfied
        assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-05, atol=1e-08)

        # Verify that the terminal state KKT conditions is satisfied
        assert jnp.allclose(dLdXs[-1], 0.0,
                            rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"

        # Verify that all KKT conditions are satisfied
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)
    
    def test_kkt_optimal_tvr(self):
        """test KKT conditions for LQR optimality"""
        # Setup the LQR problem
        lqr = self.lqr._replace(
            a=jnp.ones(self.dims["TN"]), 
            r=0.5*jnp.tile(self.x_targ, (self.dims["N"][0], 1)).T[:-1],
                                )
        Xs_dir, Us_dir, Lambs_dir = solve_lqr(LQRParams(self.x0, lqr))
        # Exercise the KKT function
        dLdXs, dLdUs, dLdLambs = kkt(LQRParams(self.x0, lqr), Xs_dir, Us_dir, Lambs_dir)
        # Plot the KKT residuals
        if OUTPUT_FIGS:
            fig = _plot_kkt(Xs_dir, Us_dir, Lambs_dir, dLdXs, dLdUs, dLdLambs)
            fig.savefig(f"{FIG_DIR}/seq_tvr_lqr_kkt.png")
            close()            
        # Verify that the average KKT conditions are satisfied
        assert jnp.allclose(jnp.mean(jnp.abs(dLdUs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdXs)), 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(jnp.mean(jnp.abs(dLdLambs)), 0.0, rtol=1e-05, atol=1e-08)

        # Verify that the terminal state KKT conditions is satisfied
        assert jnp.allclose(dLdXs[-1], 0.0,
                            rtol=1e-05, atol=1e-08), "Terminal X state not satisfied"

        # Verify that all KKT conditions are satisfied
        assert jnp.allclose(dLdUs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdXs, 0.0, rtol=1e-05, atol=1e-08)
        assert jnp.allclose(dLdLambs, 0.0, rtol=1e-05, atol=1e-08)
    
    def tearDown(self):
        """Destruct test class"""
        print("Running tearDown method...")


if __name__ == "__main__":
    unittest.main()
