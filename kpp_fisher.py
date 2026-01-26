from functools import reduce
import numpy as np

from src.utils import tensor_mult_matrix, derivative_matrix, boundary_matrix, encoding

def solve_KPP_fisher(deg, deg_out, coeffs, regBC, xzs):
    D, r = coeffs
    xs, ys = regBC
    xz1, xz2 = xzs
    N1 = tensor_mult_matrix.N1_matrix(deg,deg_out)
    GT = derivative_matrix.chebyshev_diff_matrix(deg)
    Bx = boundary_matrix.zero_value_boundary_matrix(deg,xs)
    Bt = boundary_matrix.zero_value_boundary_matrix(deg,-1)
    In = np.eye(deg+1)

    Bc1 = boundary_matrix.zero_value_boundary_matrix(deg,xz1)
    Bc2 = boundary_matrix.zero_value_boundary_matrix(deg,xz2)

    term1 = reduce(np.kron, [Bx,In,Bt,GT])/ys
    term2 = reduce(np.kron, [Bx,GT@GT,Bt,In])*D/ys
    term3 = reduce(np.kron, [Bx,In,Bt,In])/ys
    term4 = reduce(np.kron, [In,In,In,In])
    A = np.kron(N1,N1) @ (term1 - term2 - r*(term3-term4))
    B1 = np.kron(N1,N1) @ (reduce(np.kron, [Bx,Bc1,Bt,In])/ys)
    B2 = np.kron(N1,N1) @ (reduce(np.kron, [Bx,Bc2,Bt,In])/ys)

    H = A.T@A + B1.T@B1 + B2.T@B2

    eigvals, eigvecs = np.linalg.eigh(H)
    psi_sol = eigvecs[:, 0]
    print("Spectral gap:", eigvals[1] - eigvals[0])
    return psi_sol

def evaluate_KPP_fisher(deg, deg_out, regBC, psi_sol, x, t):
    xs, ys = regBC
    N1 = tensor_mult_matrix.N1_matrix(deg,deg_out)
    Bx = boundary_matrix.zero_value_boundary_matrix(deg,xs)
    Bt = boundary_matrix.zero_value_boundary_matrix(deg,-1)
    In = np.eye(deg+1)

    A = np.kron(N1,N1) @ (reduce(np.kron, [Bx,In,Bt,In])/ys)
    tau_x = encoding.chebyshev_encoding(deg=deg_out, x=x)
    tau_t = encoding.chebyshev_encoding(deg=deg_out, x=t)
    tau = np.kron(tau_x,tau_t)

    return np.dot(tau,A@psi_sol)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 3
    deg = 2**n - 1
    deg_out = 2**(n+1) - 1
    coeffs = (-1, 1)

    initCond = lambda x: np.cos(np.pi*x/2)*np.cos(np.pi*x/2)
    regBC = (0,initCond(0))
    xzs = (-1,1)

    psi_sol = solve_KPP_fisher(deg,deg_out,coeffs,regBC,xzs)
    appr_sol = lambda x,t: evaluate_KPP_fisher(deg,deg_out,regBC,psi_sol,x,t)
    s_eta = regBC[1]/appr_sol(0,-1)

    # Plot the solution
    plt.figure(figsize=(8,6))

    true_init = []
    appr_init = []
    xs = np.linspace(-1,1,100)
#     ts = np.linspace(-1,1,100)
#     X, T = np.meshgrid(xs,ts)
#     Z = appr_sol(X,T)
    for x in xs:
        true_init.append(initCond(x))
        appr_init.append(s_eta * appr_sol(x,-1))

    plt.plot(xs, appr_init, c='red', label=r'$u^*_{Q}(x,-1)$')
    plt.plot(xs, true_init, '--', label=r'$u_{true}(x,-1)$')
    plt.title(f"Solution to KPP-Fisher: n={n}")
    plt.xlabel("x")
    plt.ylabel("u(x,-1)")
    plt.legend()
    plt.grid()


#     mesh = plt.pcolormesh(X,T,Z, shading='auto', cmap='turbo')
#     plt.colorbar(mesh, label='u(x,t)')
#     plt.xlabel('Space (x)')
#     plt.ylabel('Time (t)')
#     plt.title('Exact Solution of KPP-Fisher Equation')

    plt.show()
