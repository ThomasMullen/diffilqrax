from functools import partial
from jax import random
import jax.numpy as np
from jax.scipy.linalg import block_diag
# import wandb

##s5 from https://github.com/lindermanlab/S5
from s5.train_helpers import create_train_state, reduce_lr_on_plateau,\
    linear_warmup, cosine_annealing, constant_lr, train_epoch, validate
from s5.dataloading import Datasets
from s5.seq_model import BatchClassificationModel, RetrievalModel
from s5.ssm import init_S5SSM
from s5.ssm_init import make_DPLR_HiPPO
from s5.layers import SequenceLayer
from s5.seq_model import StackedEncoderModel
import argparse
from s5.utils.util import str2bool
from diffilqrax import ilqr, parallel_ilqr
import chex
from diffilqrax import lqr
from diffilqrax.typs import (
    iLQRParams,
    LQR,
    LQRParams,
    System,
    ParallelSystem,
    ModelDims,
    Theta,
)
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, NamedTuple
"""Test functions in diffilqrax/ilqr.py"""
import unittest
from typing import Any
from pathlib import Path
from os import getcwd
import chex
import jax
from jax import Array
import jax.random as jr
import jax.numpy as jnp
import numpy as onp
from matplotlib.pyplot import subplots, close, style

from diffilqrax.utils import keygen, initialise_stable_dynamics
from diffilqrax import ilqr, parallel_ilqr
from diffilqrax.parallel_ilqr import parallel_forward_lin_integration_ilqr
from diffilqrax import lqr
from diffilqrax.typs import (
    iLQRParams,
    LQR,
    LQRParams,
    System,
    ModelDims,
    Theta,
    ParallelSystem
)

jax.config.update('jax_default_device', jax.devices('cpu')[0])
jax.config.update("jax_enable_x64", True)  # double precision
jax.config.update('jax_disable_jit', False)


def phi(x):
    return jnp.tanh(x)
##Setting up the S5 model
args = {
    "dir_name": './cache_dir',
    "dataset": 'mnist-classification',
    "n_layers": 6,
    "d_model": 3,
    "ssm_size_base": 16,
    "blocks": 8,
    "C_init": "trunc_standard_normal",
    "discretization": "zoh",
    "mode": "pool",
    "activation_fn": "half_glu1",
    "conj_sym": True,
    "clip_eigs": False,
    "bidirectional": False,
    "dt_min": 0.001,
    "dt_max": 0.1,
    "prenorm": True,
    "batchnorm": True,
    "bn_momentum": 0.95,
    "bsz": 64,
    "epochs": 100,
    "early_stop_patience": 1000,
    "ssm_lr_base": 1e-3,
    "lr_factor": 1,
    "dt_global": False,
    "lr_min": 0,
    "cosine_anneal": True,
    "warmup_end": 1,
    "lr_patience": 1000000,
    "reduce_factor": 1.0,
    "p_dropout": 0.0,
    "weight_decay": 0.05,
    "opt_config": "standard",
    "jax_seed": 1919
}



ssm_size = args["ssm_size_base"]
block_size = int(ssm_size / args["blocks"])

Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
block_size = block_size // 2
ssm_size = ssm_size // 2

Lambda = Lambda[:block_size]
V = V[:, :block_size]
Vc = V.conj().T

# If initializing state matrix A as block-diagonal, put HiPPO approximation
# on each block
Lambda = (Lambda * np.ones((args["blocks"], block_size))).ravel()
V = block_diag(*([V] * args["blocks"]))
Vinv = block_diag(*([Vc] * args["blocks"]))

init_ssm = init_S5SSM(H=args["d_model"],
                             P=ssm_size,
                             Lambda_re_init=Lambda.real,
                             Lambda_im_init=Lambda.imag,
                             V=V,
                             Vinv=Vinv,
                             C_init=args["C_init"],
                             discretization=args["discretization"],
                             dt_min=args["dt_min"],
                             dt_max=args["dt_max"],
                             conj_sym=args["conj_sym"],
                             clip_eigs=args["clip_eigs"],
                             bidirectional=args["bidirectional"])


import jax


dropout = 0
rng = jax.random.PRNGKey(0)
m = 3
horizon = 100
s5_model =  StackedEncoderModel(init_ssm, d_model = args["d_model"], n_layers = args["n_layers"], training=False)
init_rng, dropout_rng = jax.random.split(rng, num=2)
variables = s5_model.init({"params": init_rng,
                        "dropout": dropout_rng},
                        np.ones((horizon, m)), integration_timesteps=None) #length x H 
    
#model = StackedEncoderModel(ssm, d_model = d_model, n_layers = 1)
#np.ones((bsz, seq_len, in_dim))

# Print the shapes of the parameters after initialization
output = s5_model.apply(variables, np.ones((horizon, m)), integration_timesteps=0)


@jax.vmap
def diag_binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return jnp.matmul(A_j, A_i), jnp.matmul(A_j, b_i) + b_j

from jax.nn.initializers import lecun_normal, normal
from s5.ssm_init import init_CV, init_VinvB, init_log_steps, trunc_standard_normal
from s5.ssm import discretize_zoh, discretize_bilinear
from flax import linen as nn
class S5Layer(NamedTuple):
    Lambda_re: jnp.ndarray
    Lambda_im: jnp.ndarray
    B_bar: jnp.ndarray
    C_tilde: jnp.ndarray
    D: jnp.ndarray
    eff_Ks: jnp.ndarray
    
    
class S5Model():
    def __init__(self, num_layers):
        self.num_layers = num_layers 
        self.conj_sym = False
        self.clip_eigs = True
        self.blocks = 8
        self.H= 8 #d_model,
        self.P= 8 #ssm_size,
        self.blocks = num_layers#default = 2 blocks
        self.discretization = "zoh"
        self.key = 4
        self.dt_min = 0.001
        self.dt_max = 0.1
        self.layer_params = None
        self.layer_params_feedback = None
        self.lr = 0.001
        self.step_rescale = 1.0
        self.zero_im = True
        ##for now do it with 2 layers, then will actually use the flax stuff
        
    def initialize_layer(self,key):
        ssm_size = self.P
        # determine the size of initial blocks
        block_size = int(ssm_size / self.blocks)
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T
        Lambda = (Lambda * np.ones((self.blocks, block_size))).ravel()
        V = block_diag(*([V] * self.blocks))
        Vinv = block_diag(*([Vc] * self.blocks))
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
           Dierctly taken from the S5 repo https://github.com/lindermanlab/S5/tree/main/s5
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        Lambda_re = Lambda.real
        if self.zero_im :
            Lambda_im = np.zeros_like(Lambda_re)
        else : 
            Lambda_im = Lambda.imag
        if self.clip_eigs:
            Lambda = np.clip(Lambda_re, None, -1e-4) #+ 1j * Lambda_im
        else:
            Lambda = Lambda_re #+ 1j * Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        key, subkey = jr.split(key)
        B = init_VinvB(B_init, subkey, B_shape, Vinv)
        B_tilde =  B[..., 0] #+ 1j * B[..., 1]
        key, subkey = jr.split(key)
        C = jr.normal(key, shape =  (self.H, self.P, 2))
        C_tilde = C[..., 0] #+ 1j * C[..., 1]
        key, subkey = jr.split(subkey)
        D = jr.normal(key, shape = (self.H,))
        log_step = jnp.linspace(self.dt_min, self.dt_max, self.P)
        step = self.step_rescale * np.exp(log_step)

        # Discretize
        if self.discretization in ["zoh"]:
            Lambda_bar, B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            Lambda_bar, B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))
        return S5Layer(eff_Ks = jnp.zeros((self.P,self.P)), Lambda_re = Lambda_re, Lambda_im = Lambda_im, B_bar = B_bar.real, C_tilde = C_tilde.real, D = D)

    def initialize_params(self, key):
        keys = jax.random.split(key, self.num_layers) ##need to put the lengths somewhere but for now fixed
        self.layer_params = [self.initialize_layer(k) for k in keys] 
    
    def apply_single_layer_full(self, layer_params, input_sequence):
        Lambda_bar = layer_params.Lambda_re #+ 1j * layer_params.Lambda_im
        Bu_elements = jax.vmap(lambda u: layer_params.B_bar @ u)(input_sequence)
        Lambda_elements = jnp.tile(jnp.diag(Lambda_bar), (input_sequence.shape[0],1,1))
        
        #jnp.diag(Lambda_bar) * np.ones((input_sequence.shape[0],
         #                                       Lambda_bar.shape[0])) - layer_params.eff_Ks
        
        _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
        ##other optiosn : birectional, conj sym -> not using those yet
        return jax.vmap(lambda x: (layer_params.C_tilde @ phi(x)))(xs), xs
    #jax.vmap(lambda x: (layer_params.C_tilde @ x).real)(xs), xs
    
    def apply_single_layer_step(self, layer_params, xl_t, ul_t):
        Lambda_bar = layer_params.Lambda_re #+ 1j * layer_params.Lambda_im
        Lambda_bar = jnp.diag(Lambda_bar)
        nx = Lambda_bar @ xl_t + layer_params.B_bar @ ul_t
        otpt = layer_params.C_tilde @ phi(nx)
        #_, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
        ##other optiosn : birectional, conj sym -> not using those yet
        return nx, otpt#.real
    
    def single_s5_step(self, xs_t, u0_t):
        ##go through the layers
        u = u0_t
        nxs = []
        for i, layer_prms in enumerate(self.layer_params):
            x = xs_t[i*self.P:(i+1)*self.P]
            nx, o = self.apply_single_layer_step(layer_prms, x, u)
            nxs.append(nx)
            u = o
        return jnp.concatenate(nxs), u
    
    def apply_full(self, us):
        ##go through the layers
        u = us
        nxs = []
        for layer_prms in self.layer_params:
            u, nx = self.apply_single_layer_full(layer_prms, u)
            nxs.append(nx)
        return jnp.concatenate(nxs, axis = 1), u

    def apply_full_feedback(self, us, Kxs):
        ##go through the layers
        u = us
        nxs = []
        for i, layer_prms in enumerate(self.layer_params):
            B = layer_prms.B_bar
            Lambda_re = layer_prms.Lambda_re 
            Kx_l = Kxs[...,i*self.P:(i+1)*self.P]
            #(100, 16, 64) (32, 16) (32,)
            #new_layer_0 = layer_params_feedback[0]._replace(Lambda_re =jax.vmap(lambda l, b, k: l + b@k, in_axes = (None,None,0))(Lambda_re, B, Kx))
            #layer_prms_feedback = [new_layer_0 if i == 0 else layer_params_feedback[i] for i in np.arange(len(layer_params_feedback))]
            #s5model.layers_params_feedback = layer_prms_feedback
            ##we lose the nice diagonal property when we do this
            eff_Ks = jax.vmap(lambda b, k: b@k, in_axes = (None,0))(B, Kx_l)
            new_layer_prms = layer_prms._replace(eff_Ks = eff_Ks)
            u, nx = self.apply_single_layer_full(new_layer_prms, u)
            nxs.append(nx)
        return jnp.concatenate(nxs, axis = 1), u
##Setting up the iLQR problem and tests


class TestS5(unittest.TestCase):
    """Test LQR dimensions and data structures"""

    def setUp(self):
        """Setup LQR problem"""
        s5model = S5Model(num_layers = 2)
        s5model.initialize_params(key = jax.random.PRNGKey(2))
        dt = 0.1
        Uh = jnp.array([[1, dt], [-1 * dt, 1 - 0.5 * dt]])
        Wh = jnp.eye(3)*dt #jnp.array([[0, 0,1], [1, 0]]) * dt
        Q = jnp.eye(2)
        key = jr.PRNGKey(0)
        # initialise params
        self.theta = Theta(Uh=Uh, Wh=Wh, sigma=jnp.zeros((2)), Q=Q)
        self.params = iLQRParams(x0=jnp.zeros(2*s5model.P), theta=self.theta)
        x_tgt = jnp.arange(2*s5model.P)/2*s5model.P
        def cost(t: int, x: Array, u: Array, theta: Theta):
            return jnp.sum((x.squeeze()[-1] - x_tgt.squeeze())**2) + 0.001*jnp.sum(u**2)

        def costf(x: Array, theta: Theta):
            # return jnp.sum(jnp.abs(x))
            return 0*jnp.sum(x**2)

        def dynamics(t: int, x: Array, u: Array, theta: Theta):
            nx, _ = s5model.single_s5_step(x, u)
            return nx#.real
        def parallel_dynamics(model, theta, us, a):
            xs, os = s5model.apply_full(us)
            return jnp.concatenate([theta.x0[None], xs])
        def parallel_dynamics_feedback(model, theta, us, a, Kxs):
            xs, os = s5model.apply_full_feedback(us, Kxs)
            return jnp.concatenate([theta.x0[None], xs])
            # layer_params_feedback = s5model.layer_params
            # B = s5model.layer_params[0].B_bar
            # Lambda_re = s5model.layer_params[0].Lambda_re
            # print(Kx.shape, B.shape, Lambda_re.shape)
            # #(100, 16, 64) (32, 16) (32,)
            # new_layer_0 = layer_params_feedback[0]._replace(Lambda_re =jax.vmap(lambda l, b, k: l + b@k, in_axes = (None,None,0))(Lambda_re, B, Kx))
            # layer_prms_feedback = [new_layer_0 if i == 0 else layer_params_feedback[i] for i in np.arange(len(layer_params_feedback))]
            # s5model.layers_params_feedback = layer_prms_feedback
            # xs, os = s5model.apply_full_feedback(us)
            # return jnp.concatenate([theta.x0[None], xs.real])
        
        m = s5model.H
        self.dims = ModelDims(horizon=100, n=2*s5model.P, m=m, dt=dt)
        self.model = System(cost, costf, dynamics, self.dims)
        self.parallel_model = ParallelSystem(
            self.model, parallel_dynamics, parallel_dynamics_feedback
        )
        key = jr.PRNGKey(seed=234)
        key, skeys = keygen(key, 3)
        self.Us_init = 0. * jr.normal(
            next(skeys), (self.model.dims.horizon, self.model.dims.m)
        )
        # define linesearch parameters
        self.ls_kwargs = {
        "beta": 0.8,
        "max_iter_linesearch": 10,
        "tol": 0.1,
        "alpha_min": 0.0001,
        }


    def test_pilQR_solver(self):
        """test ilqr solver with integrater dynamics"""
        # setup
        fig_dir = Path(Path(getcwd()), "fig_dump", "s5")
        fig_dir.mkdir(exist_ok=True)
        # exercise
        (Xs_stars, Us_stars, Lambs_stars), converged_cost, cost_log = parallel_ilqr.pilqr_solver(
            self.parallel_model,
            self.params,
            self.Us_init,
            max_iter=70,
            convergence_thresh=1e-8,
            alpha_init=1.0,
            verbose=True,
            use_linesearch=True,
            **self.ls_kwargs,
        )
        #difference when Us_init is not 0...
        (Xs_stars_ilqr, Us_stars_ilqr, _), converged_cost, cost_log = ilqr.ilqr_solver(
            self.model,
            self.params,
            self.Us_init,
            max_iter=70,
            convergence_thresh=1e-8,
            alpha_init=1.0,
            verbose=True,
            use_linesearch=True,
            **self.ls_kwargs,
        )
        print(Xs_stars_ilqr.shape)
        fig, ax = subplots(2, 2, sharey=True)
        ax[0, 1].plot(Us_stars)
        ax[0, 0].plot(Xs_stars)
        ax[0, 1].set(title="U (parallel)")
        ax[0, 0].set(title="X (parallel)")
        ax[1, 0].plot(Xs_stars_ilqr)
        ax[1, 1].plot(Us_stars_ilqr)
        ax[1, 1].set(title="U (normal)")
        ax[1, 0].set(title="X (normal)")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/s5_pilqr_solver.png")
        # NOTE: tolerance is high revisit
        chex.assert_trees_all_close(Xs_stars, Xs_stars_ilqr, rtol=6e-01)
        
     ##add speed test of pilqr   
if __name__ == "__main__":
    unittest.main()
    
    

# nxs, output = s5model.full_s5_step(Us_init)
# print(nxs[1].shape)
# x0 = [jnp.zeros(s5model.P) for _ in range(s5model.num_layers)]
# nx, o = s5model.single_s5_step(jnp.concatenate(x0), Us_init[0])

#(Xs_init, _), _= parallel_ilqr.ilqr_simulate(model, Us_init, params)
# (Xs_stars, Us_stars, Lambs_stars), total_cost, cost_log = parallel_ilqr.pilqr_solver(
#             model,
#             params,
#             Us_init,
#             max_iter=80,
#             convergence_thresh=1e-13,
#             alpha_init=1.,
#             verbose=True,
#             use_linesearch=True,
#             **ls_kwargs,
#         )

# from matplotlib import pyplot as plt
# fig, axes = plt.subplots(3,1)
# axes[0].plot(Us_init)
# axes[1].plot(Us_stars)
# axes[2].plot(Xs_stars)
