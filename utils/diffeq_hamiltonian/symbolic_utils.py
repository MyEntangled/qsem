"""
Shared symbolic utilities for the diffeq_hamiltonian package.

Strategy F — Factored Taylor Expansion:
  For separable expressions like sin(J₀·ξ₁+c₀)·cos(J₁·ξ₂+c₁), expanding each
  univariate factor independently in 1D and combining via outer product avoids the
  O(N^d) combinatorial blowup of sequential multivariate sp.series().
"""

import sympy as sp


def factored_series_poly_dict(expr, xi_syms, truncated_order):
    """
    Expand *expr* as a polynomial in *xi_syms* and return the monomial dictionary
    ``{powers_tuple: coeff_expr}``, where *coeff_expr* may still contain other
    symbols (e.g. Jacobian, center symbols).

    When the expression (or an additive sub-term) is a product of factors that
    each depend on at most **one** ξ variable, each factor is Taylor-expanded
    independently in 1D and the results are combined via an outer product of
    polynomial coefficients — dramatically cheaper than sequential multivariate
    ``sp.series()``.

    Non-separable sub-terms (e.g. ``sin(ξ₁ + ξ₂)``) fall back to the standard
    sequential expansion.

    Parameters
    ----------
    expr : sp.Expr
        Expression to expand (may be non-polynomial in some ξ variables).
    xi_syms : tuple of sp.Symbol
        Reference-domain variables (ξ₁, ξ₂, …).
    truncated_order : int or None
        Taylor truncation order. If None and the expression is non-polynomial,
        a ValueError is raised.

    Returns
    -------
    dict
        ``{(p₁, p₂, …): coeff_expr}`` — same format as ``sp.Poly.as_dict()``.
    """
    xi_syms = tuple(xi_syms)

    # Fast path: already polynomial in all ξ — no series expansion needed
    if all(expr.is_polynomial(v) for v in xi_syms):
        return sp.Poly(expr, *xi_syms).as_dict()

    # Truncation order is required for non-polynomial expressions
    if truncated_order is None:
        raise ValueError(
            "truncated_order must be specified for non-polynomial coefficients."
        )

    # Split into additive terms and process each independently
    combined = {}
    for term in sp.Add.make_args(expr):
        term_dict = _expand_term(term, xi_syms, truncated_order)
        for powers, coeff in term_dict.items():
            if powers in combined:
                combined[powers] = combined[powers] + coeff
            else:
                combined[powers] = coeff

    return combined


def _expand_term(term, xi_syms, truncated_order):
    """Expand a single multiplicative term, using factored 1D expansion when separable."""
    xi_set = set(xi_syms)
    n_vars = len(xi_syms)

    # If the term is already polynomial in all ξ, no expansion needed
    if all(term.is_polynomial(v) for v in xi_syms):
        return sp.Poly(term, *xi_syms).as_dict()

    # Split into multiplicative factors
    factors = sp.Mul.make_args(term)

    # Group factors by which ξ variables they depend on
    scalar_part = sp.S.One          # factors independent of all ξ
    var_groups = {}                  # var_index -> accumulated expression
    non_separable = False

    for factor in factors:
        deps = factor.free_symbols & xi_set
        if not deps:
            # Factor doesn't involve any ξ (e.g. J, c, numeric constants)
            scalar_part *= factor
        elif len(deps) == 1:
            # Univariate in exactly one ξ
            var = next(iter(deps))
            var_idx = xi_syms.index(var)
            if var_idx in var_groups:
                var_groups[var_idx] *= factor
            else:
                var_groups[var_idx] = factor
        else:
            # Factor depends on multiple ξ → not separable
            non_separable = True
            break

    if non_separable:
        # Fall back to sequential multivariate expansion
        return _sequential_series(term, xi_syms, truncated_order)

    # === Separable path: expand each univariate group in 1D ===
    per_var_polys = {}  # var_idx -> {power_int: coeff_expr}

    for var_idx, group_expr in var_groups.items():
        var = xi_syms[var_idx]
        if not group_expr.is_polynomial(var):
            expanded = sp.series(group_expr, var, 0, truncated_order).removeO()
            expanded = sp.expand(expanded)
        else:
            expanded = group_expr
        # Extract 1D polynomial: {(p,): coeff} -> {p: coeff}
        poly_1d = sp.Poly(expanded, var).as_dict()
        per_var_polys[var_idx] = {powers[0]: coeff for powers, coeff in poly_1d.items()}

    # Variables that don't appear in any factor get identity: {0: 1}
    for i in range(n_vars):
        if i not in per_var_polys:
            per_var_polys[i] = {0: sp.S.One}

    # Outer product of 1D polynomial dictionaries
    combined = {(): scalar_part}
    for i in range(n_vars):
        new_combined = {}
        for existing_powers, existing_coeff in combined.items():
            for power, coeff in per_var_polys[i].items():
                new_powers = existing_powers + (power,)
                new_coeff = existing_coeff * coeff
                if new_powers in new_combined:
                    new_combined[new_powers] = new_combined[new_powers] + new_coeff
                else:
                    new_combined[new_powers] = new_coeff
        combined = new_combined

    return combined


def _sequential_series(expr, xi_syms, truncated_order):
    """Fall back: sequential multivariate Taylor expansion (current behavior)."""
    result = expr
    for v in xi_syms:
        if not result.is_polynomial(v):
            result = sp.series(result, v, 0, truncated_order).removeO()
    result = sp.expand(result)
    return sp.Poly(result, *xi_syms).as_dict()
