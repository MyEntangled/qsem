import sympy as sp
import numpy as np


class ElementalMultiVariateEquationParser:
    """
    Parse multivariate (possibly nonlinear) differential expressions into grouped terms.

    Target canonical form per additive term:
        c(x) * f(x)^k * Π_j ( ∂^{alpha_j} f(x) )^{m_j}

    where:
      - k is the power of the function f itself
      - each derivative factor is identified by a multi-index alpha_j over the variables
      - cross-derivative products are supported (multiple distinct derivative atoms per term)

    Output:
      terms_dict : dict
        Keys are (deriv_signature, k) where:
          - k : int (power of f)
          - deriv_signature : tuple of (alpha_tuple, power_int) sorted canonically
            e.g. for 2D variables (x,y):
              ( ((1,0), 2), ((0,1), 1) )  represents  (f_x)^2 * (f_y)^1
              ( ((1,1), 1), )            represents  f_xy
              ( () , )                   represents no derivatives
        Values are sympy expressions for coefficient c(x).
    """

    def __init__(self):
        return

    def parse_equation(self, diff_eq, func_sym):
        """
        Expand the expression and group additive terms by (k, derivative-multiindex signature).

        Parameters
        ----------
        diff_eq : sympy.Expr
            Differential expression (LHS or any expression).
        func_sym : sympy.Expr
            The function application, e.g. f(x,y) (NOT sp.Function('f')).

        Returns
        -------
        dict
            Mapping (k, deriv_signature) -> coefficient c(x).
        """
        terms_dict = {}

        # Expand derivatives and algebra
        expanded_eq = sp.expand(diff_eq.doit())

        # Split into additive terms
        for term in sp.Add.make_args(expanded_eq):
            self._process_term(func_sym, term, terms_dict)

        return terms_dict

    def _process_term(self, func_sym, term, terms_dict):
        """
        Analyze a single additive term and accumulate it into terms_dict.
        """
        # Power map: base -> exponent (best-effort canonical)
        powers_map = term.as_powers_dict()

        # Identify all derivative atoms that involve func_sym
        deriv_atoms = [d for d in term.atoms(sp.Derivative) if d.has(func_sym)]

        # Determine function power k (f may appear as a factor)
        # If f does not appear, k=0.
        k = int(powers_map.get(func_sym, 0))

        # Build derivative signature: multi-index alpha + power m
        deriv_factors = []
        denom = (func_sym ** k)

        if deriv_atoms:
            # We want DISTINCT derivative atoms (cross products allowed).
            # Use set for uniqueness; keep as list for deterministic ordering later.
            unique_derivs = list(set(deriv_atoms))

            for d_atom in unique_derivs:
                alpha = self._derivative_multiindex(d_atom, func_sym)
                m = powers_map.get(d_atom, 1)
                # SymPy exponents can be Integers; enforce int where possible
                if not (m.is_integer and m.is_nonnegative):
                    raise ValueError(
                        f"Non-integer or negative power for derivative atom {d_atom}: {m}"
                    )
                m_int = int(m)

                if m_int == 0:
                    continue

                deriv_factors.append((alpha, m_int))
                denom *= (d_atom ** m_int)

            # Canonical ordering: lexicographic in alpha, then by power
            deriv_signature = tuple(sorted(deriv_factors, key=lambda t: (t[0], t[1])))

            # Coefficient is what remains after dividing out f^k and derivative factors
            coeff = sp.simplify(term / denom)

        else:
            # No derivatives: derivative signature empty
            deriv_signature = tuple()
            if k > 0:
                coeff = sp.simplify(term / (func_sym ** k))
            else:
                coeff = term

        key = (deriv_signature, k)
        terms_dict[key] = terms_dict.get(key, 0) + coeff
        return terms_dict

    def _derivative_multiindex(self, deriv_atom: sp.Derivative, func_sym):
        """
        Convert a SymPy Derivative(f(x1,...,xd), ...) into a multi-index alpha over the
        variables in func_sym.args order.

        Example:
          func_sym = f(x,y,z)
          Derivative(f(x,y,z), x, x, z)  -> alpha = (2,0,1)
        """
        if not deriv_atom.has(func_sym):
            raise ValueError("deriv_atom does not involve the provided func_sym.")

        # Variables defining the coordinate order
        vars_order = list(func_sym.args)
        if not vars_order:
            raise ValueError("func_sym must be a function application with arguments, e.g. f(x,y).")

        # SymPy stores differentiation variables in deriv_atom.variables (can repeat)
        counts = [0] * len(vars_order)
        for v in deriv_atom.variables:
            try:
                idx = vars_order.index(v)
            except ValueError as e:
                raise ValueError(
                    f"Derivative variable {v} not found among func_sym arguments {vars_order}."
                ) from e
            counts[idx] += 1

        return tuple(counts)

    def transform(self, terms_dict, vars, lo, hi):
        num_vars = len(vars)
        if not (num_vars == len(lo) == len(hi)):
            raise ValueError("'lo' and 'hi' should provide endpoints for the variables.")

        hi, lo = np.array(hi), np.array(lo)

        # Use sp.symbols (plural) to create a tuple of distinct symbols
        # xi_1:num_vars+1 creates (xi_1, xi_2, ..., xi_n)
        local_vars = sp.symbols(f"xi_1:{num_vars + 1}")
        local_terms_dict = {}

        # Calculate Jacobian and inverse Jacobian per dimension
        J = (hi - lo) / 2
        J_inv = 2 / (hi - lo)

        # Create the mapping for each variable: x_i = J_i * xi_i + center_i
        var_maps = [J[i] * local_vars[i] + (hi[i] + lo[i]) / 2 for i in range(num_vars)]

        for (deriv_signature, k), coeff in terms_dict.items():
            # 1. Substitute spatial variables x_i -> x_i(xi_i)
            subs_map = list(zip(vars, var_maps))
            new_coeff = coeff.subs(subs_map)

            # 2. Apply Chain Rule Scaling Factor
            total_scale_factor = 1.0

            # deriv_signature: ( ((alpha_tuple), power), ... )
            for item in deriv_signature:
                # Handle the empty signature case ()
                if not item or not isinstance(item, tuple) or len(item) < 2:
                    continue

                alpha_tuple, power = item

                for i, order in enumerate(alpha_tuple):
                    if order > 0:
                        # Scaling factor for partial derivative (d^n/dx^n)^p is (1/J)^(n*p)
                        total_scale_factor *= J_inv[i] ** (order * power)

            # 3. Combine and simplify
            transformed_coeff = sp.expand(new_coeff * total_scale_factor)

            # 4. Store
            local_terms_dict[(deriv_signature, k)] = transformed_coeff

        return local_terms_dict, local_vars


# --- Example usage (remove in production) ---
if __name__ == "__main__":
    x, y = sp.symbols("x y")
    f = sp.Function("f")(x, y)

    fx = sp.Derivative(f, x)
    fy = sp.Derivative(f, y)
    fxy = sp.Derivative(f, x, y)

    expr = (x**2) * (fx**2) * (fy) + (1 - x*y) * fxy - 3 * f**2 + sp.sin(x + y)

    parser = ElementalMultiVariateEquationParser()
    terms = parser.parse_equation(expr, f)

    for key, c in terms.items():
        sig, k = key
        print(f"key: k={k}, sig={sig}  -> coeff: {c}")


    local_terms = parser.transform(terms, vars=[x, y], lo=[0, 0], hi=[1, 1])
    print("\nTransformed Terms:")
    for key, c in local_terms[0].items():
        sig, k = key
        print(f"key: k={k}, sig={sig}  -> coeff: {c}")