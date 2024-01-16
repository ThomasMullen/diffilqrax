"""Back-tracking line search garentees convergences for non-linear systems"""
import jax
import jax.lax as lax
import jax.numpy as np



# recursive linesearch
def linesearch():

    def continue_search(vals: Tuple, expected_dC: Callable):
        cost, cost_old, _, _, alpha = vals
        p = np.abs((cost_old - cost) / expected_dC(alpha))
        return p > p_thresh

    def body_fun(vals, gamma: float = gamma):
        old_cost, _, Xs, Us, alpha = vals
        alpha *= gamma
        (new_Xs, new_Us), new_cost = rollout(Xs, Us, alpha)
        return new_cost, old_cost, new_Xs, new_Us, alpha
    
    # structure of while loop represented as a scan
    def while_loop(
        # continue_search: Callable[[float, float, float], bool],
        # body_fun: Callable,
        init_val: Tuple[Any],
        gamma: float = 0.5,
        max_iter: int = 20,
        p_thresh: float = 1e-3,
    ):
        # initialise values
        rollout = partial(ddp_rollout, model=model, params=params, Ks=Ks)
        # init_val = None


        def loop(carry):
            cond, vals = carry
            new_vals = lax.cond(cond, body_fun, lambda x: x, vals)
            return (continue_search(new_vals), new_vals), new_vals

        (converged, final_state), _ = lax.scan(
            loop, init=(continue_search(init_val), init_val), length=max_iter
        )

        return converged, final_state

    pass

def linesearch():
    # Run the linear search until convergence
    def while_loop(cond_fun, body_fun, init_val, maxiter):
        def loop_body(carry):
            cond, state = carry
            new_state = lax.cond(cond, body_fun, lambda x: x, state)
            return (cond, new_state)

        _, final_state = lax.scan(loop_body, (cond_fun(init_val), init_val), length=maxiter)
        return final_state

    # conditional check: if the updated is significantly less that updated
    def cond_fun(state):
        return np.sum(state) < 100

    def body_fun(state):
        return state + 1

    initial_state = np.array(0)
    result = while_loop(cond_fun, body_fun, initial_state)
    
    
    
    pass