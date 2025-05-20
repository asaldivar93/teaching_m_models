from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from cobra.util import create_stoichiometric_matrix

if TYPE_CHECKING:
    from cobra import Model, Reaction
    from pandas import DataFrame


def find_blocked_mets(model: Model, tol: float = 1e-9) -> list[list[str]]:
    """Find orphan and deadend metabolites.

    Parameters
    ----------
    model: user supplied model
    tol: numerical tolrance (default: 1e-9)

    Returns
    -------
    orphans: Metabolites that are consumed but not produced
    deadends: Metabolites that are produced but not produced

    References
    ----------
    Beguerisse-Díaz, M., Bosque, G., Oyarzún, D. et al. Flux-dependent graphs
    for metabolic networks. npj Syst Biol Appl 4, 32 (2018).
    https://doi.org/10.1038/s41540-018-0067-y

    """
    m = len(model.reactions)

    s_df: DataFrame = create_stoichiometric_matrix(model, "DataFrame")

    s_np = create_stoichiometric_matrix(model, "dense")
    i_np = np.identity(m)
    r_ = []  # Reversibility vector

    for identifier in s_df.columns:
        rxn: Reaction = model.reactions.get_by_id(identifier)
        if rxn.reversibility:
            r_.append(1)
        else:
            r_.append(0)

    # 2-m dimensional stoichiometric matrix
    s2m_ = np.dot(
        np.block([s_np, -s_np]),
        np.block([[i_np, np.zeros([m, m])], [np.zeros([m, m]), np.diag(r_)]]),
    )

    # Production Stoichiometric matrix
    s2m_p = (1 / 2) * (np.abs(s2m_) + s2m_)
    # Consumption Stoichiometric matrix
    s2m_c = (1 / 2) * (np.abs(s2m_) - s2m_)

    # Metabolites that are consumed but not produced
    orphans: list[str] = s_df.loc[s2m_p.sum(axis=1) < tol, :].index.to_list()
    # Metabolites that are produced but not produced
    deadends: list[str] = s_df.loc[s2m_c.sum(axis=1) < tol, :].index.to_list()

    return [orphans, deadends]


def build_normilized_flow_graph(model, tol=1e-9):
    """Method implemented from:
    Beguerisse-Díaz, M., Bosque, G., Oyarzún, D. et al. Flux-dependent graphs
    for metabolic networks. npj Syst Biol Appl 4, 32 (2018).
    https://doi.org/10.1038/s41540-018-0067-y
    """
    m = len(model.reactions)
    n = len(model.metabolites)

    S = create_stoichiometric_matrix(model, "DataFrame")

    S_ = create_stoichiometric_matrix(model, "dense")
    Im_ = np.identity(m)
    r_ = []

    for id in S.columns:
        rxn = model.reactions.get_by_id(id)
        if rxn.reversibility:
            r_.append(1)
        else:
            r_.append(0)

    S2m_ = np.dot(
        np.block([S_, -S_]),
        np.block([[Im_, np.zeros([m, m])], [np.zeros([m, m]), np.diag(r_)]]),
    )
    S2m_p = (1 / 2) * (np.abs(S2m_) + S2m_)
    S2m_c = (1 / 2) * (np.abs(S2m_) - S2m_)

    W_p = np.linalg.pinv(np.diag(np.dot(S2m_p, np.ones(2 * m))))
    W_c = np.linalg.pinv(np.diag(np.dot(S2m_c, np.ones(2 * m))))
    S2m_p.shape
    normalized_flow_graph = (
        np.dot(np.dot(W_p, S2m_p).transpose(), np.dot(W_c, S2m_c)) / n
    )
    p = normalized_flow_graph.sum()

    if abs(1 - p) > tol:
        print(f"Sum of probabilities ({p}) below tolerance level {tol}")
        print("Remove blocked metabolites and reactions from the model")

    return normalized_flow_graph, S2m_p, S2m_c


def build_mass_flow_graph(item):
    solution = item[0]
    S2m_p = item[1]
    S2m_c = item[2]

    v_ = solution
    v2m_ = (1 / 2) * np.block([np.abs(v_) + v_, np.abs(v_) - v_])
    jv_ = np.dot(S2m_p, v2m_)

    V_ = np.diag(v2m_)
    Jvi_ = np.linalg.pinv(np.diag(jv_))

    mass_flow_graph = np.dot(
        np.dot(np.dot(S2m_p, V_).transpose(), Jvi_),
        np.dot(S2m_c, V_),
    )
    graph = nx.from_numpy_array(mass_flow_graph, create_using=nx.DiGraph)
    pagerank = pd.DataFrame(nx.pagerank(graph, alpha=0.90), index=["pagerank"])

    return pagerank
