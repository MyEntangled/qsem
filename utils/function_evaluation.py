from src.utils import encoding, multiply_matrix
import numpy as np

def evaluate_sem_encoding(psi_sol: np.ndarray,
                          deg: int,
                          deg_out: int,
                          endpoints: np.ndarray,
                          x_eval_list: np.ndarray,
                          scaling_factor: float = 1.0) -> np.ndarray:
    assert endpoints.ndim in [2,3]
    num_elements = endpoints.shape[0]
    assert len(psi_sol) == num_elements * (deg + 1)

    lo = endpoints[:, 0]
    hi = endpoints[:, 1]

    # Reshape psi_sol to (num_elements, deg+1)
    psi_sol = psi_sol.reshape((num_elements, deg + 1))
    # Normalize psi_el for each element
    #psi_sol /= np.linalg.norm(psi_sol, axis=1, keepdims=True)

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
        #print(tau_e.shape, np.linalg.norm(tau_e), np.linalg.norm(psi_e))
        #print(M1 @ psi_e)
        if deg_out != deg:
            f_e = np.dot(tau_e, M1 @ psi_e)
        else:
            f_e = np.dot(tau_e, psi_e)

        f_eval_list.append(f_e)

    f_eval_list = np.array(f_eval_list) * scaling_factor
    return f_eval_list


