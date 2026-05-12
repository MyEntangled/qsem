import sympy as sp
import numpy as np


class VectorMultiVariateEquationParser:
    """
    Parse systems of multivariate differential expressions for vector-valued functions.

    Target canonical form per additive term:
        c(x) * Π_i ( f_{idx_i}(x) )^{k_i} * Π_j ( ∂^{alpha_j} f_{idx_j}(x) )^{m_j}
    """

    def __init__(self):
        pass

    def parse_equation(self, diff_eq, func_syms):
        """
        Parameters
        ----------
        diff_eq : sympy.Expr
            The differential expression.
        func_syms : list of sympy.Expr
            List of function applications, e.g., [f(x,y), g(x,y)].
        """
        terms_dict = {}
        expanded_eq = sp.expand(diff_eq.doit())

        for term in sp.Add.make_args(expanded_eq):
            self._process_term(func_syms, term, terms_dict)

        return terms_dict

    def _process_term(self, func_syms, term, terms_dict):
        powers_map = term.as_powers_dict()

        # 1. Identify powers of the base functions: (comp_idx, power)
        f_powers = []
        denom = sp.Integer(1)
        for idx, f_sym in enumerate(func_syms):
            k = int(powers_map.get(f_sym, 0))
            if k > 0:
                f_powers.append((idx, k))
                denom *= (f_sym ** k)

        f_signature = tuple(sorted(f_powers))

        # 2. Identify derivative atoms: (comp_idx, alpha_tuple, power)
        deriv_factors = []
        # Find all derivatives that involve ANY of the functions in the vector
        deriv_atoms = [d for d in term.atoms(sp.Derivative)
                       if any(d.has(f) for f in func_syms)]

        if deriv_atoms:
            unique_derivs = list(set(deriv_atoms))
            for d_atom in unique_derivs:
                # Find which component this derivative belongs to
                comp_idx = next(i for i, f in enumerate(func_syms) if d_atom.has(f))
                f_sym = func_syms[comp_idx]

                alpha = self._derivative_multiindex(d_atom, f_sym)
                m = powers_map.get(d_atom, 1)

                if not (m.is_integer and m.is_nonnegative):
                    raise ValueError(f"Non-integer power for derivative: {m}")

                m_int = int(m)
                if m_int > 0:
                    deriv_factors.append((comp_idx, alpha, m_int))
                    denom *= (d_atom ** m_int)

        # Canonical ordering for derivatives
        deriv_signature = tuple(sorted(deriv_factors, key=lambda t: (t[0], t[1], t[2])))

        # 3. Extract coefficient
        coeff = sp.simplify(term / denom)

        key = (deriv_signature, f_signature)
        terms_dict[key] = terms_dict.get(key, 0) + coeff

    def _derivative_multiindex(self, deriv_atom, func_sym):
        vars_order = list(func_sym.args)
        counts = [0] * len(vars_order)
        for v in deriv_atom.variables:
            idx = vars_order.index(v)
            counts[idx] += 1
        return tuple(counts)

    def transform(self, terms_dict, vars, lo, hi):
        """
        Maps variables to local domain [-1, 1] and scales derivatives.
        """
        num_vars = len(vars)
        hi, lo = np.array(hi), np.array(lo)
        local_vars = sp.symbols(f"xi_1:{num_vars + 1}")

        J_inv = 2 / (hi - lo)
        var_maps = {v: (hi[i] - lo[i]) / 2 * local_vars[i] + (hi[i] + lo[i]) / 2
                    for i, v in enumerate(vars)}

        local_terms_dict = {}

        for (deriv_sig, f_sig), coeff in terms_dict.items():
            # Substitute spatial variables in coefficient
            new_coeff = coeff.subs(var_maps)

            # Apply Chain Rule Scaling
            # Derivative scaling: (d^n/dx^n) introduces (1/J)^n
            total_scale_factor = 1.0
            for comp_idx, alpha_tuple, power in deriv_sig:
                for i, order in enumerate(alpha_tuple):
                    if order > 0:
                        total_scale_factor *= J_inv[i] ** (order * power)

            transformed_coeff = sp.expand(new_coeff * total_scale_factor)
            local_terms_dict[(deriv_sig, f_sig)] = transformed_coeff

        return local_terms_dict, local_vars


# --- Example Usage for Vector Case ---
if __name__ == "__main__":
    x, y = sp.symbols("x y")
    # Define vector function u = [f(x,y), g(x,y)]
    f = sp.Function("f")(x, y)
    g = sp.Function("g")(x, y)
    funcs = [f, g]

    # Example: x*f_x + g*g_y = 0
    expr = x * sp.Derivative(f, x) + g * sp.Derivative(g, y) + f*g

    parser = VectorMultiVariateEquationParser()
    terms = parser.parse_equation(expr, funcs)

    print("Parsed Vector Terms:")
    for (d_sig, f_sig), c in terms.items():
        # d_sig format: (component_index, alpha, power)
        # f_sig format: (component_index, power)
        print(f"Coeff: {c} | f_powers: {f_sig} | derivatives: {d_sig}")