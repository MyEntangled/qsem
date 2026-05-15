import numpy as np
from src.utils.encoding import chebyshev_encoding
from functools import reduce

def chebyshev_int_state(deg,dim):
    I = np.array([1/(1-(i+j)**2) + 1/(1-(i-j)**2) if (i+j)%2 == 0 else 0
         for i in range(deg+1)
         for j in range(deg+1)])
    tau = chebyshev_encoding(deg=deg, x=1)
    iota = np.kron(tau,tau)*I
    if dim == 1:
        return iota
    else:
        M = reduce(np.kron, [iota.reshape(deg+1,deg+1)] * dim)
        return M.flatten()

def constract_iota_state(iota, mats):
    shape_in = [m.shape[0] for m in mats]
    res = iota.reshape(shape_in)
    for i in range(len(mats)):
        res = np.tensordot(res,mats[i],axes=([0],[0]))
    return res.flatten()

def construct_boundary_matrix(deg,xs,ys,fs,xb,yb,g,lam=1):
    n = deg+1
    N = n*n
    tau_xs = chebyshev_encoding(deg=deg, x=xs)
    tau_ys = chebyshev_encoding(deg=deg, x=ys)

    B = np.zeros((N,N))
    if xb is not None:
        tau_xb = chebyshev_encoding(deg=deg, x=xb)

        ## Term 1
        B_view = B.reshape(n,n,n,n)
        block = fs*fs*np.outer(tau_xb,tau_xb)
        for i in range(n):
            B_view[:,i,:,i] = block

        if g is not None:
        ## Term 2
            v_target = np.outer(tau_xb,g).flatten()
            v_global = np.outer(tau_xs,tau_ys).flatten()
            term2 = fs * np.outer(v_target,v_global)
            B -= (term2+term2.T)

        ## Term 3
            term3 = np.sum(g*g) * np.outer(v_global,v_global)
            B += term3

    else:
        tau_yb = chebyshev_encoding(deg=deg, x=yb)

        ## Term 1
        block = fs*fs*np.outer(tau_yb,tau_yb)
        for i in range(n):
            B[i*n:(i+1)*n,i*n:(i+1)*n] = block

        if g is not None:
        ## Term 2
            v_target = np.outer(g,tau_yb).flatten()
            v_global = np.outer(tau_xs,tau_ys).flatten()
            term2 = fs * np.outer(v_target,v_global)
            B -= (term2+term2.T)

        ## Term 3
            term3 = np.sum(g*g) * np.outer(v_global,v_global)
            B += term3

    return lam*B

def test_basis_matrix(deg):
    Z = np.eye(deg+1, deg-1, k=0) - np.eye(deg+1, deg-1, k=-2)
    Z[0,0] = np.sqrt(2)
    return Z
