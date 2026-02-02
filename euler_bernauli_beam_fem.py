import numpy as np
import matplotlib.pyplot as plt

class EulerBernoulliBeamFEM:
    def __init__(self, L, n_elements, E, I):
        self.L = L
        self.n_elements = n_elements
        self.E = E
        self.I = I
        
        self.nodes = np.linspace(-L/2, L/2, n_elements + 1)
        self.n_nodes = len(self.nodes)
        self.element_length = L / n_elements
        self.n_dof = 2 * self.n_nodes  # 2 DOFs per node (displacement w, rotation theta)
        
        self.K = np.zeros((self.n_dof, self.n_dof))
        self.F = np.zeros(self.n_dof)
        
    def assemble_stiffness_matrix(self):
        # Using Linear Timoshenko Beam Elements with Reduced Integration
        # This employs linear shape functions for both displacement and rotation (C0 continuity),
        # approximating the Euler-Bernoulli beam by using a large shear stiffness penalty.
        # This is the standard "linear element" formulation for beams.
        
        le = self.element_length
        
        # 1. Bending Stiffness (Exact integration for constant Strain/linear Theta)
        # U_b = 0.5 * integral(EI * (theta')^2 dx)
        # theta' = (th2 - th1)/le
        # Stiffness coeff: EI/le
        kb_val = self.E * self.I / le
        
        # 2. Shear Stiffness (1-point Reduced Integration)
        # U_s = 0.5 * integral(kGA * (w' - theta)^2 dx)
        # We assume a large penalty for shear stiffness to enforce w' ~ theta (Euler-Bernoulli Limit)
        # kGA (Shear Rigidity) approx factor * EI / le^2
        shear_penalty_factor = 1e5
        kGA = shear_penalty_factor * (self.E * self.I) / (le**2)
        
        # Integrated value (Integration weight = le)
        # ks_coeff = kGA * le
        ks_val = kGA * le
        
        # Local stiffness matrix for [w1, th1, w2, th2]
        k_local = np.zeros((4, 4))
        
        # Bending contributions (Only affects th1 (idx 1) and th2 (idx 3))
        # [  1  -1 ]
        # [ -1   1 ]
        k_local[1, 1] += kb_val
        k_local[1, 3] -= kb_val
        k_local[3, 1] -= kb_val
        k_local[3, 3] += kb_val
        
        # Shear contributions
        # Strain B-matrix at midpoint: [ -1/le, -0.5, 1/le, -0.5 ]
        # K_shear = ks_val * B.T @ B
        B_s = np.array([-1/le, -0.5, 1/le, -0.5])
        k_shear = ks_val * np.outer(B_s, B_s)
        
        k_local += k_shear
        
        for i in range(self.n_elements):
            # Element DOFs: [w_i, th_i, w_{i+1}, th_{i+1}]
            # Global indices: 2*i, 2*i+1, 2*i+2, 2*i+3
            indices = [2*i, 2*i+1, 2*i+2, 2*i+3]
            
            for row in range(4):
                for col in range(4):
                    self.K[indices[row], indices[col]] += k_local[row, col]
                    
    def apply_uniform_load(self, q):
        le = self.element_length
        # Consistent load vector for linear elements:
        # work = integral(q * w dx)
        # w is linear, so we split load equally to nodes: q*le/2
        # No moment load generation from distributed force in linear elements
        
        f_local = np.array([q * le / 2, 0, q * le / 2, 0])
        
        for i in range(self.n_elements):
            indices = [2*i, 2*i+1, 2*i+2, 2*i+3]
            for j in range(4):
                self.F[indices[j]] += f_local[j]

    def apply_boundary_conditions(self, bcs):
        """
        Apply boundary conditions.
        bcs: list of tuples (node_index, dof_type, value)
             dof_type: 0 for displacement, 1 for rotation
        """
        # Direct stiffness modification method
        for node_idx, dof_type, val in bcs:
            dof_idx = 2 * node_idx + dof_type
            
            self.K[dof_idx, :] = 0
            self.K[:, dof_idx] = 0
            self.K[dof_idx, dof_idx] = 1
            self.F[dof_idx] = val

    def solve(self):
        return np.linalg.solve(self.K, self.F)

    def plot_results(self, U, analytical_func=None):
        displacements = U[0::2]
        # Rotations are U[1::2]
        
        x_fine = np.linspace(0, self.L, 100)
        
        plt.figure(figsize=(10, 6))
        # Linear elements: plotting points with lines is the exact representation
        plt.plot(self.nodes, displacements, 'o-', label='FEM Displacement (Linear Elements)')
        
        if analytical_func:
            y_analytical = analytical_func(x_fine)
            plt.plot(x_fine, y_analytical, 'r--', label='Analytical')
            
        plt.xlabel('Position x')
        plt.ylabel('Deflection w(x)')
        plt.title('Euler-Bernoulli Beam (via Timoshenko Linear Elements)')
        plt.legend()
        plt.grid()
        plt.savefig('beam_deflection_FEM.png')
        plt.show()

def run_example():
    L = 2.0
    n_elements = 20 # Increased element count for linear elements convergence
    E = 1.0
    I = 1.0
    q0 = -1.0 
    
    fem = EulerBernoulliBeamFEM(L, n_elements, E, I)
    fem.assemble_stiffness_matrix()
    fem.apply_uniform_load(q0)
    
    # Boundary Conditions: Clamped at both ends
    bcs = [
        (0, 0, 0.0), # Node 0, Disp
        (0, 1, 0.0), # Node 0, Rot
        (n_elements, 0, 0.0), # Node n, Disp
        (n_elements, 1, 0.0)  # Node n, Rot
    ]
    
    fem.apply_boundary_conditions(bcs)
    
    U = fem.solve()
    
    # Analytical solution for clamped-clamped beam length L under load q
    def analytical(x):
        # w(x) = (q x^2 (L-x)^2) / (24 EI)
        return (q0 * x**2 * (L - x)**2) / (24 * E * I)
        
    fem.plot_results(U, analytical_func=analytical)

if __name__ == "__main__":
    run_example()
