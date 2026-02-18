from src.utils import encoding, multiply_matrix
import numpy as np

def evaluate_sem_encoding(psi_sol: np.ndarray,
                          deg: int,
                          deg_out: int,
                          endpoints: np.ndarray,
                          x_eval_list: np.ndarray,
                          scaling_factor: float = 1.0) -> np.ndarray:

    assert endpoints.ndim == 2
    num_elements = endpoints.shape[0]
    assert len(psi_sol) == num_elements * (deg + 1)

    lo = endpoints[:, 0]
    hi = endpoints[:, 1]

    # Reshape psi_sol to (num_elements, deg+1)
    psi_sol = psi_sol.reshape((num_elements, deg + 1))

    # Evaluate at the provided x_list
    f_eval_list = []
    for x_eval in x_eval_list:
        # Find the element containing x
        e_s = None
        for e in range(num_elements):
            if lo[e] <= x_eval <= hi[e]:
                e_s = e
                break
        if e_s is None:
            raise ValueError(f"Point x={x_eval} is not found in the mesh.")
        xi_eval = (2 * x_eval - (lo[e_s] + hi[e_s])) / (hi[e_s] - lo[e_s])  # Map to [-1, 1]

        # Evaluate the solution at x using the Chebyshev expansion for element e_s
        lo, hi = endpoints[e_s]
        psi_e = psi_sol[e_s]

        tau_e = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval)

        M1 = multiply_matrix.M_x_power(deg, p=0, deg_out=deg_out)

        if deg_out != deg:
            f_e = np.dot(tau_e, M1 @ psi_e)
        else:
            f_e = np.dot(tau_e, psi_e)

        f_eval_list.append(f_e)

    f_eval_list = np.array(f_eval_list) * scaling_factor
    return f_eval_list


def evaluate_multivar_sem_encoding(psi_sol: np.ndarray,
                          deg: int,
                          deg_out: int,
                          endpoints: np.ndarray,
                          coords_eval_list: np.ndarray,
                          scaling_factor: float = 1.0) -> np.ndarray:
    assert endpoints.ndim == 3
    num_elements = endpoints.shape[0]
    num_vars = endpoints.shape[2]
    assert len(psi_sol) == num_elements * (deg + 1)** num_vars

    lo = endpoints[:, 0, :]
    hi = endpoints[:, 1, :]

    # Reshape psi_sol to (num_elements, (deg+1)^num_vars)
    psi_sol = psi_sol.reshape((num_elements, (deg + 1) ** num_vars))

    # Precompute 1D lift matrix deg -> deg_out
    M1_1d = multiply_matrix.M_x_power(deg, p=0, deg_out=deg_out)

    # Build the full tensor-product lift matrix only if needed
    lift = None
    if deg_out != deg:
        lift = None
        for _ in range(num_vars):
            lift = M1_1d if lift is None else np.kron(lift, M1_1d)

    # Evaluate at the provided coordinate list
    f_eval_list = []
    for coords_eval in coords_eval_list:
        coords_eval = np.asarray(coords_eval, dtype=float)
        if coords_eval.shape != (num_vars,):
            raise ValueError(
                f"Each evaluation point must have shape ({num_vars},), got {coords_eval.shape}."
            )

        # Find the element containing coords_eval
        e_s = None
        for e in range(num_elements):
            if np.all(lo[e, :] <= coords_eval) and np.all(coords_eval <= hi[e, :]):
                e_s = e
                break
        if e_s is None:
            raise ValueError(f"Point coords={tuple(coords_eval)} is not found in the mesh.")

        # Map to reference coordinates xi in [-1, 1]^d
        xi_eval = (2.0 * coords_eval - (lo[e_s, :] + hi[e_s, :])) / (hi[e_s, :] - lo[e_s, :])

        # Tensor-product Chebyshev encoding tau(xi) = kron_v tau_v(xi_v)
        tau = None
        for v in range(num_vars):
            tau_v = encoding.chebyshev_encoding(deg=deg_out, x=xi_eval[v])
            tau = tau_v if tau is None else np.kron(tau, tau_v)

        psi_e = psi_sol[e_s]

        if deg_out != deg:
            f_e = np.dot(tau, lift @ psi_e)
        else:
            f_e = np.dot(tau, psi_e)

        f_eval_list.append(f_e)

    f_eval_list = np.array(f_eval_list) * scaling_factor
    return f_eval_list
