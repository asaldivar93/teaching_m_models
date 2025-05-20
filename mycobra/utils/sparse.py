from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from cobra.flux_analysis.helpers import normalize_cutoff
from optlang.symbolics import Zero

if TYPE_CHECKING:
    from cobra import Model


def find_leaks(model: Model, tol: float = 1e-6, keep_boundaries: bool = False):
    """Find all the posible leaks of a model.

    max_y,v ||y||_0
    s.t. S*v - y = 0
         0 <= y for met in metabolites
         lb <= v <= ub

    Since ||y||_0 is an NP-Hard problem, this is re casted
    as a LP using L1-norm as an approximation

    Parameters
    ----------
    model: a cobra model
    tol: numerical tolerance
    keep_boundaries: default False

    Returns
    -------
    leaks: a list of leaked metabolites.

    References
    ----------
    [ref]

    """
    sparse_model = deepcopy(model)
    prob = sparse_model.problem

    # keep only internal reactions
    if not keep_boundaries:
        for rxn in sparse_model.reactions:
            if rxn.boundary:
                sparse_model.remove_reactions(rxn)

    # empty all constraints
    for const in sparse_model.constraints:
        sparse_model.remove_cons_vars(const)

    met_vars = []
    for met in sparse_model.metabolites:
        # Create auxiliary variable y
        met_var = prob.Variable(
            f"aux_{met.id}",
            lb=0,
        )
        # Create constraint
        const = prob.Constraint(
            Zero,
            name=met.id,
            lb=0,
            ub=0,
        )
        sparse_model.add_cons_vars([met_var, const])

        rxn_coeffs = []
        # Get stoichiometrich coefficients
        # for all reactions of met
        for rxn in met.reactions:
            coeff = rxn.metabolites[met]
            rxn_coeffs.append([rxn.forward_variable, coeff])
            rxn_coeffs.append([rxn.reverse_variable, -coeff])
        # Add auxiliary var to constraint
        rxn_coeffs.append([met_var, -1])
        # Add constraint to model
        rxn_coeffs = dict(rxn_coeffs)
        sparse_model.constraints.get(met.id).set_linear_coefficients(rxn_coeffs)

        met_vars.extend([met_var])

    zero_cutoff = normalize_cutoff(sparse_model, tol)
    # Set objective to max(sum(met_vars))
    sparse_model.objective = prob.Objective(Zero, direction="max")
    sparse_model.objective.set_linear_coefficients({o: 1 for o in met_vars})

    sparse_model.optimize()
    leaks = []
    # if y_i > 0 met_i is a leak in the model
    for var in sparse_model.variables:
        if "aux" in var.name and abs(var.primal) > zero_cutoff:
            leaks.append([var.name.replace("aux_", ""), var.primal])

    return leaks


def find_leak_mode(
    model,
    leaks=[],
    cutoff_mult=1,
    flux_mult=1,
    keep_boundaries=False,
):
    """The minimal set of reactions needed for production
    of leak metabolites (leak_mode) cand be found by min of
    the following program:

    min_y,v ||v||_0
    s.t. S*v - y = 0
         y >= 1 for met in leaks
         y >= 0 for met not in leaks
         lb <= v <= ub

    To approximate ||v||_0, we add an indicator variable z
    for every reaction and minimized L_2 norm of vector z in
    the following program:

    min_z,y,v (sum_r(z_r**2))**(1/2)
    s.t. S*v - y = 0
         v - z = 0
         z unbounded
         y >= 1 for met in leaks
         y >= 0 for met not in leaks
         lb <= v <= ub

    Parameters
    ----------
    sparse_model: cobra.core.Model
        The cobra model (with aux vars and constrains) to find leaks from
    leaks: list
        A list of leaking metabolites and fluxes

    Returns
    -------
    leak_modes: dict
        A dictionary of active reactions for every leak metabolite

    References
    ----------
    [ref]

    """
    sparse_model = deepcopy(model)
    prob = sparse_model.problem
    zero_cutoff = normalize_cutoff(sparse_model, None)
    objective = Zero
    rxn_vars_and_cons = []

    # keep only internal reactions
    if not keep_boundaries:
        for rxn in sparse_model.reactions:
            if rxn.boundary:
                sparse_model.remove_reactions(rxn)

    # empty all constraints
    for const in sparse_model.constraints:
        sparse_model.remove_cons_vars(const)

    met_vars = []
    for met in sparse_model.metabolites:
        # Create auxiliary variable y
        met_var = prob.Variable(
            f"aux_{met.id}",
            lb=0,
        )
        # Create constraint
        const = prob.Constraint(
            Zero,
            name=met.id,
            lb=0,
            ub=0,
        )
        sparse_model.add_cons_vars(
            [met_var, const],
        )

        rxn_coeffs = []
        # Get stoichiometrich coefficients
        # for all reactions of met
        for rxn in met.reactions:
            coeff = rxn.metabolites[met]
            rxn_coeffs.append([rxn.forward_variable, coeff])
            rxn_coeffs.append([rxn.reverse_variable, -coeff])
        # Add auxiliary var to constraint
        rxn_coeffs.append(
            [met_var, -1],
        )
        # Add constraint to model
        rxn_coeffs = dict(rxn_coeffs)
        sparse_model.constraints.get(met.id).set_linear_coefficients(rxn_coeffs)

        met_vars.extend(
            [met_var],
        )

    # Add z variables and associated constraints
    for rxn in sparse_model.reactions:
        rxn_var = prob.Variable(f"rxn_{rxn.id}")

        const = prob.Constraint(
            rxn.forward_variable + rxn.reverse_variable - rxn_var,
            name=f"rxn_{rxn.id}",
            lb=0,
            ub=0,
        )
        rxn_vars_and_cons.extend(
            [rxn_var, const],
        )
        objective += rxn_var**2

    sparse_model.add_cons_vars(
        rxn_vars_and_cons,
    )

    # Add objective. Since solvers only take linear or quadratic objectives
    # the objective function is left as sum_r(z_r**2)
    sparse_model.objective = prob.Objective(
        objective,
        direction="min",
    )

    # find active reactions for every leak individually
    leak_modes = {}
    print(zero_cutoff)
    for leak, flux in leaks:
        rxns_in_mode = []

        met_var = sparse_model.variables.get(f"aux_{leak}")
        met_var.lb = flux
        sol = sparse_model.optimize()
        met_var.lb = 0

        rxns_in_mode = [
            [rxn.id, sol.fluxes[rxn.id]]
            for rxn in sparse_model.reactions
            if abs(sol.fluxes[rxn.id]) >= cutoff_mult * zero_cutoff
        ]
        leak_modes[leak] = sol
        print(leak, sol.status, len(rxns_in_mode))

    return leak_modes
