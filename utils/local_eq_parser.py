import sympy as sp


class ElementalEquationParser:
    def __init__(self):
        #self.terms_dict = {}
        #self.f_sym = None
        #self.x_sym = None
        #self.xi_sym = sp.Symbol('xi')  # Reference domain variable
        return

    def parse_equation(self, diff_eq, func_sym):
        """
        Expands the equation and groups terms by (order n, deriv_power m, func_power k).

        Refers to the form: c(x) * f^k * (d^n f / dx^n)^m
        """
        # self.f_sym = func
        # self.x_sym = var
        # self.terms_dict = {}
        terms_dict = {}

        # 1. Expand Derivatives and Algebra
        # doit(): applies product rule d/dx(f^2) -> 2*f*f'
        # expand(): separates additive terms
        expanded_eq = sp.expand(diff_eq.doit())

        # 2. Iterate over additive terms
        args = sp.Add.make_args(expanded_eq)

        for term in args:
            self._process_term(func_sym, term, terms_dict)

        return terms_dict

    def _process_term(self, func_sym, term, terms_dict):
        """
        Analyzes a single additive term to find (n, m, k).
        """
        # Get the power map of the term directly
        # e.g., "5 * x * f**2 * (f')**3" -> {5:1, x:1, f:2, f':3}
        powers_map = term.as_powers_dict()

        # 1. Identify Derivative Order (n)
        # Find all derivatives of f in this term
        derivs = [d for d in term.atoms(sp.Derivative) if d.has(func_sym)]

        if derivs:
            # We assume no cross-terms (e.g., f' * f''), so there is only one unique derivative atom
            # or multiple of the same order. We pick the highest just in case.
            deriv_atom = max(derivs, key=lambda d: d.derivative_count)
            n = deriv_atom.derivative_count

            # 2. Get Power of Derivative (m)
            m = powers_map.get(deriv_atom, 1)

            # 3. Get Power of Function f (k)
            # The function f might appear as a coefficient (quasi-linear)
            k = powers_map.get(func_sym, 0)

            # 4. Extract Coefficient c(x)
            # c(x) = term / ( (f^(n))^m * f^k )
            # We divide out the parts we identified
            denom = (deriv_atom ** m) * (func_sym ** k)
            coeff = sp.simplify(term / denom)

        else:
            # Case: No derivatives found (n=0)
            # The term is likely f^p or just c(x)
            n = 0
            m = 0  # No derivative operator

            if term.has(func_sym):
                # It is a reaction term like c(x) * f^3
                # We map this to k=total_power
                k = powers_map.get(func_sym, 1)

                # c(x) = term / f^k
                coeff = sp.simplify(term / (func_sym ** k))
            else:
                # Pure forcing term (no f)
                k = 0
                coeff = term

        # Store in Dictionary
        # Key: (Order n, Deriv Power m, Func Power k)
        #terms_dict = self._add_coeff((n, m, k), coeff, terms_dict)

        if (n,m,k) in terms_dict:
            terms_dict[(n,m,k)] += coeff
        else:
            terms_dict[(n,m,k)] = coeff
        return terms_dict


    # def _add_coeff(self, key, val, terms_dict):
    #     if key in terms_dict:
    #         terms_dict[key] += val
    #     else:
    #         terms_dict[key] = val
    #     return terms_dict

    def transform(self, terms_dict, var_sym, a:float, b:float):
        """
        Transforms terms from Physical x in [a,b] to Reference xi in [-1,1].

        Logic:
        Term c(x) * f^k * (d^n f/dx^n)^m
        becomes
        [c(x(xi)) * (InvJac)^(n*m)] * f^k * (d^n f/dxi^n)^m
        """
        local_var_sym = sp.Symbol('xi')
        local_terms_dict = {}

        J = (b - a) / 2
        J_inv = 2 / (b - a)
        var_map = J * local_var_sym + (b + a) / 2

        for (n, m, k), coeff_expr in terms_dict.items():
            # 1. Substitute x -> x(xi) in the coefficient
            # This handles the spatial dependency
            new_coeff = coeff_expr.subs(var_sym, var_map)

            # 2. Apply Chain Rule Scaling Factor
            # Each derivative order n contributes (1/J)^n
            # This derivative is raised to power m, so total scale is (1/J)^(n*m)
            # Note: f^k contributes no scale factor
            scale_factor = J_inv ** (n * m)

            # 3. Combine
            transformed_coeff = sp.expand(new_coeff) * scale_factor

            # 4. Store (Structure is invariant)
            local_terms_dict[(n, m, k)] = transformed_coeff

        return local_terms_dict, local_var_sym


if __name__ == "__main__":
    # 1. Define Symbols
    x = sp.Symbol('x')
    f = sp.Function('f')(x)

    # 2. Define a complex Quasi-Linear Equation
    # Example: Burger's-like or Porous medium equation terms
    # Term 1: d/dx( f^2 * f' ) -> expanded to f^2*f'' + 2*f*(f')^2
    # Term 2: - f^3 (Reaction)
    # Term 3: + sin(x) (Forcing)

    # Let's construct it manually to test the parser logic clearly:
    # LHS = f^2 * f'' + 2 * f * (f')^2 - f^3 + sp.sin(x)

    # Note: I am writing the expanded form directly to test categorization.
    # The parser .doit() handles unexpanded d/dx inputs too.
    d1 = f.diff(x)
    d2 = f.diff(x, 2)

    #lhs = (f ** 2 * d2) + (2 * f * d1 ** 2) - f ** 3 + sp.sin(x)
    #d / dx(x ^ 2 * u') + (x^2-1)*(u') ^ 2 - u ^ 3 + sin(x)
    lhs = sp.Derivative(x ** 2 * sp.Derivative(f, x), x) + (x**2-1) * (sp.Derivative(f, x) ** 2) - f ** 3 + sp.sin(x) + (f ** 2 * d2)
    print(f"Equation LHS: {lhs}")

    # 3. Parse
    parser = ElementalEquationParser()
    terms = parser.parse_equation(lhs, f)
    print("Parsed Terms:")
    print(terms)

    local_terms, xi = parser.transform(terms, x, a=0, b=1)
    print("Transformed Terms (Reference Domain):")
    print(local_terms)

    # 4. Display Results
    print(f"{'Key (n, m, k)':<15} | {'Term Type':<25} | {'Coefficient c(x)'}")
    print("-" * 70)

    # Sort for cleaner output
    sorted_keys = sorted(terms.keys(), key=lambda t: (t[0], t[1], t[2]), reverse=True)

    for (n, m, k) in sorted_keys:
        coeff = terms[(n, m, k)]

        # Generate description
        if n == 0 and k == 0:
            desc = "Forcing"
        elif n == 0:
            desc = f"Reaction f^{k}"
        else:
            desc = f"f^{k} * (f^({n}))^{m}"

        print(f"({n}, {m}, {k}){' ':6} | {desc:<25} | {coeff}, {coeff.is_polynomial(x)}")