import numpy as np
from utils import derivative_matrix, encoding

def complex_step_derivative(func, h=1e-20):
    """
    Computes the
    """

def zero_value_boundary_matrix(deg, x_z):
    """
    Computes the boundary matrix B(x_z) such that:
    f(x_z) = 0 --> sqrt(eta) <tau(x)|_d B(x_z) |psi> = 0
    Specifically, B(x_z) = sqrt(d+1) |0><tau(x_z)|_d
    """
    d = deg
    B = np.zeros((d + 1, d + 1))
    
    # tau = np.zeros(d + 1)
    # scale = 1.0 / np.sqrt(d + 1)
    # for k in range(d + 1):
    #     if k == 0:
    #         tau[k] = scale
    #     else:
    #         tau[k] = np.cos(k * np.arccos(x_z)) * scale * np.sqrt(2)

    tau = encoding.chebyshev_encoding(deg, x_z)

    B[0, :] = tau
    B *= np.sqrt(d + 1)

    return B

def zero_derivative_boundary_matrix(deg, x_m):
    """
    Computes the boundary matrix hat{B}(x_m) such that:
    f'(x_m) = 0 --> sqrt(eta) <tau(x)|_d hat{B}(x_m) |psi> = 0
    Specifically, hat{B}(x_m) = sqrt(d+1) |0><tau(x_z)|_d G^T
    """

    B = zero_value_boundary_matrix(deg, x_m)
    B_hat = B @ derivative_matrix.chebyshev_diff_matrix(deg, deg_out=None)

    return B_hat

def regular_value_boundary_matrix(deg, x_s, y_s):
    """
    Computes the boundary matrix D(x_s) such that:
    f(x_s) = y_s --> sqrt(eta) <tau(x)|_d D^{(0)}(x_s) |psi> = 1
    Specifically, D^{(0)}(x_s) = B(x_s) / y_s
    """
    B = zero_value_boundary_matrix(deg, x_s)
    D0 = B / y_s
    return D0

def regular_derivative_boundary_matrix(deg, x_s, t_s):
    """
    Computes the boundary matrix D^{(1)}(x_s) such that:
    f'(x_s) = t_s --> sqrt(eta) <tau(x)|_d D^{(1)}(x_s) |psi> = 1
    Specifically, D^{(1)}(x_s) = hat{B}(x_s) / t_s
    """
    B_hat = zero_derivative_boundary_matrix(deg, x_s)
    D1 = B_hat / t_s
    return D1
