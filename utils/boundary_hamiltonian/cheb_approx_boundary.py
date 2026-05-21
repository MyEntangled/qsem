from typing import Iterable, Callable

import numpy as np
from scipy.fft import dctn
import numpy.polynomial.chebyshev as cheb

def _get_cheb_coeffs_(func: Callable, degrees: int | Iterable):
    if isinstance(degrees, int):
        degrees = (degrees,) * func.__code__.co_argcount

    d = len(degrees)
    N_array = np.array(degrees) + 1

    nodes = [np.cos(np.pi * (np.arange(N) + 0.5) / N) for N in N_array]
    grid = np.meshgrid(*nodes, indexing='ij', sparse=True)

    f_values = func(*grid)
    coeffs = dctn(f_values, type=2, norm=None, workers=-1)

    scaling = np.ones_like(coeffs)
    for axis, N in enumerate(N_array):
        # We need to divide by N for k > 0.
        axis_scale = np.full(N, 1.0 / N)
        # The DC component (k=0) needs to be exactly half of the other coefficients
        axis_scale[0] = 0.5 / N

        shape = [1] * d
        shape[axis] = N

        axis_scale = axis_scale.reshape(shape)
        scaling *= axis_scale

    return coeffs * scaling

def cheb_coeffs_projector(func: Callable, degrees: int | Iterable, include_matrices: bool = True):
    """Compute the Chebyshev coefficients and optionally the orthogonal projector.
    
    If include_matrices is False, only tau_coeffs is returned (others are None).
    This avoids O(N^2) memory allocation for large basis sizes.
    """
    if isinstance(degrees, int):
        degrees = (degrees,) * func.__code__.co_argcount

    coeffs = _get_cheb_coeffs_(func, degrees)

    ## Scale coeffs according to quantum encoding.
    normalization = np.ones_like(coeffs)

    for axis in range(coeffs.ndim):
        axis_norm = np.sqrt(degrees[axis] + 1)
        axis_weights = [1.0 / axis_norm] + [np.sqrt(2.0) / axis_norm] * degrees[axis]
        axis_weights = np.array(axis_weights)

        shape = [1] * len(degrees)
        shape[axis] = coeffs.shape[axis]

        axis_weights = axis_weights.reshape(shape)
        normalization *= axis_weights

    tau_coeffs = coeffs / normalization
    tau_coeffs = tau_coeffs.flatten()

    if not include_matrices:
        return tau_coeffs, None, None

    proj = np.outer(tau_coeffs, tau_coeffs) / np.sum(tau_coeffs**2)
    orthog_proj = np.eye(len(tau_coeffs)) - proj

    return tau_coeffs, proj, orthog_proj



if __name__ == "__main__":
    # ==========================================
    # Example 1: 1D Approximation
    # ==========================================
    print("--- 1D Example ---")
    # Define a highly oscillatory 1D function
    func_1d = lambda x: np.sin(5 * x) * np.exp(x)

    # Get coefficients for a polynomial of degree 15
    coeffs_1d = _get_cheb_coeffs_(func_1d, 15)
    print(coeffs_1d.shape)

    # Evaluate at a random test point
    x_test = 0.42
    exact_val = func_1d(x_test)
    approx_val = cheb.chebval(x_test, coeffs_1d)
    print(f"Exact:  {exact_val:.6f}")
    print(f"Approx: {approx_val:.6f}\n")


    # ==========================================
    # Example 2: 2D Approximation
    # ==========================================
    print("--- 2D Example ---")

    # Define a 2D function
    func_2d = lambda x, y: np.cos(3 * x) * np.exp(-y**2)

    # Get coefficients for a polynomial of max degree 12x12
    coeffs_2d = _get_cheb_coeffs_(func_2d, (12, 12))
    print(coeffs_2d.shape)

    # Evaluate at specific test points (x, y)
    x_test_2d = np.array([-0.5, 0.1, 0.9])
    y_test_2d = np.array([0.5, -0.2, 0.8])

    exact_vals_2d = func_2d(x_test_2d, y_test_2d)
    # NumPy provides chebval2d specifically for evaluating 2D coefficient matrices
    approx_vals_2d = cheb.chebval2d(x_test_2d, y_test_2d, coeffs_2d)

    for i in range(len(x_test_2d)):
        print(f"Point ({x_test_2d[i]:.1f}, {y_test_2d[i]:.1f}) -> Exact: {exact_vals_2d[i]:.6f} | Approx: {approx_vals_2d[i]:.6f}")
    print()


    # ==========================================
    # Example 3: 3D Approximation
    # ==========================================
    print("--- 3D Example ---")
    # Define a 3D function (e.g., a Runge-like function)
    func_3d = lambda x, y, z: 1 / (1 + x**2 + y**2 + z**2)

    # Get coefficients for a polynomial of max degree 10x10x10
    coeffs_3d = _get_cheb_coeffs_(func_3d, (10, 10, 10))

    # Evaluate at a random test point
    x_test_3d, y_test_3d, z_test_3d = 0.3, -0.7, 0.5

    exact_val_3d = func_3d(x_test_3d, y_test_3d, z_test_3d)
    # NumPy provides chebval3d for 3D tensors
    approx_val_3d = cheb.chebval3d(x_test_3d, y_test_3d, z_test_3d, coeffs_3d)

    print(f"Exact:  {exact_val_3d:.6f}")
    print(f"Approx: {approx_val_3d:.6f}")


    # ==========================================
    # Testing different degree
    # ==========================================
    func_mixed = lambda x, y: np.sin(10 * x) * np.exp(-y)
    degrees_mixed = (20, 5)

    coeffs_mixed = _get_cheb_coeffs_(func_mixed, degrees_mixed)
    #print(coeffs_mixed.shape)

    x_test, y_test = 0.45, -0.2
    exact_val = func_mixed(x_test, y_test)
    approx_val = cheb.chebval2d(x_test, y_test, coeffs_mixed)

    print(f"Exact:  {exact_val:.6f}")
    print(f"Approx: {approx_val:.6f}")


    #######
    from src.utils import encoding

    def get_tau(func, degs, vars):
        for var, deg in zip(vars, degs):
            tau_var = encoding.chebyshev_encoding(deg, var)
            if 'tau' not in locals():
                tau = tau_var
            else:
                tau = np.kron(tau_var, tau)
        return tau

    func = lambda x, y: np.sin(10 * x) * np.exp(-y)
    degs = (20,5)

    _, proj, orthog_proj = cheb_coeffs_projector(func, degs)

    x_s, y_s = 0.45, -0.2
    x_test, y_test = 0.2, -0.1
    tau_s = get_tau(func, degs, (x_s, y_s))
    tau_test = get_tau(func, degs, (x_test, y_test))

    psi = np.ones(np.prod(np.array(degs) + 1))
    psi = psi / np.linalg.norm(psi)

    scaling_factor = func(x_s, y_s) / np.dot(tau_s, proj @ psi)
    scaling_factor_test = func(x_test, y_test) / np.dot(tau_test, proj @ psi)


    print(f"Scaling factor:", scaling_factor, scaling_factor_test)

    print("Overlap model:" , np.dot(tau_test, proj @ psi) * scaling_factor)
    print("Direct eval:", func(x_test, y_test))



