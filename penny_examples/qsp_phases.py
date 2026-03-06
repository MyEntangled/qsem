# Import relevant modules and methods.
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from scipy.special import iv

from pyqsp import angle_sequence, response

deg = 15
t = 8

func_cosh = lambda x: np.cosh(t*x)/np.exp(t)
coeffs = np.zeros(2*deg+1)
for i in range(deg):
    coeffs[2*i] = iv(2*i,t)/np.exp(t)
coeffs[1:] *= 2
poly_cosh = Chebyshev(coeffs)

func_sinh = lambda x: np.sinh(t*x)/np.exp(t)
coeffs = np.zeros(2*deg+1)
for i in range(deg):
    coeffs[2*i+1] = 2*iv(2*i+1,t)/np.exp(t)
poly_sinh = Chebyshev(coeffs)

# Compute full phases (and reduced phases, parity) using symmetric QSP.
(phiset_cosh, _, _) = angle_sequence.QuantumSignalProcessingPhases(
    poly_cosh,
    method='sym_qsp',
    chebyshev_basis=True)

(phiset_sinh, _, _) = angle_sequence.QuantumSignalProcessingPhases(
    poly_sinh,
    method='sym_qsp',
    chebyshev_basis=True)

np.savez('qsp_phases.npz', cosh=phiset_cosh, sinh=phiset_sinh)
print("Phases saved to qsp_phases.npz")

"""
Plot response according to full phases.
Note that `pcoefs` are coefficients of the approximating polynomial,
while `target` is the true function (rescaled) being approximated.
"""
response.PlotQSPResponse(
    phiset_cosh,
    pcoefs=poly_cosh,
    target=func_cosh,
    sym_qsp=True,
    simul_error_plot=True)

response.PlotQSPResponse(
    phiset_sinh,
    pcoefs=poly_sinh,
    target=func_sinh,
    sym_qsp=True,
    simul_error_plot=True)
