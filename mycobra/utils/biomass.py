from cobra.util import create_stoichiometric_matrix

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import networkx as nx


import cobra


def update_biomass(model, BIOMASS_RXN='BIOMASS', GAM=21, NADH=[]):
    file = '/home/alexis/UAM/Papers/Verrucomicrobia/cobra/Mfumariolicum_pic/data_files/biomass.csv'
    biomass_in_model = pd.read_csv(file, index_col='bigg')
    ATP_growth = GAM
    BM_weight = 0
    stoichiometry = {}
    
    rxn = model.reactions.get_by_id(BIOMASS_RXN)
    biomass = model.metabolites.biomass
    
    rxn.add_metabolites({met: -rxn.metabolites[met] for met in rxn.metabolites.keys()})

    for precursor in biomass_in_model.index:
        try:
            met = model.metabolites.get_by_id(precursor + '_c')
        except KeyError:
            try:
                met = model.metabolites.get_by_id(precursor + '_p')
            except KeyError:
                try:
                    met = model.metabolites.get_by_id(precursor + '_im')
                except KeyError:
                    continue
        stoichiometry[met] = - biomass_in_model.loc[precursor, 'mmol/gDWbiomass']
        BM_weight += - met.formula_weight * stoichiometry[met]
    
    # biomass molecular weight = 1g/mol
    biomas_coeff = {
        met: stoichiometry[met] * 1000 / BM_weight for met in stoichiometry.keys()
    }

    biomas_coeff['biomass'] = 1

    # Add Stoichiometry
    rxn.add_metabolites(
        biomas_coeff
    )
    rxn.upper_bound = 1000
    
    # Calculate Biomass formula from element balance
    element_balance = rxn.check_mass_balance()
    formula = ''
    for e in element_balance:
        if not e == 'charge':
            formula += str(e) + str(format(-element_balance[e], '.30f'))
    biomass.formula = formula
    BM_weight = biomass.formula_weight / 1000

    energy = {'atp_c': -ATP_growth,
              'adp_c': ATP_growth,
              'pi_c': ATP_growth,
              'h2o_c': ATP_growth}
    rxn.add_metabolites(
        energy
    )

    # Add NAD(P)H and PROTON requirements
    element_balance = rxn.check_mass_balance()
    energy = {'h_c': -element_balance['charge']}
    rxn.add_metabolites(
        energy
    )
    element_balance = rxn.check_mass_balance()
    rxn.add_metabolites(
        {'h2o_c': -element_balance['O']}
    )
    element_balance = rxn.check_mass_balance()
    rxn.add_metabolites(
        {'h_c': -element_balance['H'] / 2,
         'nadh_c': -element_balance['H'] / 2,
         'nad_c': element_balance['H'] / 2}
    )
    biomass.formula=""

    return model



def find_blocked_mets(model):
    m = len(model.reactions)

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

    S2m_ = np.dot(np.block([S_, -S_]), np.block([[Im_, np.zeros([m, m])], [np.zeros([m, m]), np.diag(r_)]]))
    S2m_p = (1 / 2) * (np.abs(S2m_) + S2m_)
    S2m_c = (1 / 2) * (np.abs(S2m_) - S2m_)

    orphans = S.loc[S2m_p.sum(axis=1) < 1e-9, :].index.to_list()
    deadends = S.loc[S2m_c.sum(axis=1) < 1e-9, :].index.to_list()

    return orphans, deadends


def build_normilized_flow_graph(model, tol=1e-9):
    """
    Method implemented from:
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

    S2m_ = np.dot(np.block([S_, -S_]), np.block([[Im_, np.zeros([m, m])], [np.zeros([m, m]), np.diag(r_)]]))
    S2m_p = (1 / 2) * (np.abs(S2m_) + S2m_)
    S2m_c = (1 / 2) * (np.abs(S2m_) - S2m_)

    W_p = np.linalg.pinv(np.diag(np.dot(S2m_p, np.ones(2 * m))))
    W_c = np.linalg.pinv(np.diag(np.dot(S2m_c, np.ones(2 * m))))
    S2m_p.shape
    normalized_flow_graph = np.dot(np.dot(W_p, S2m_p).transpose(), np.dot(W_c, S2m_c)) / n
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

    mass_flow_graph = np.dot(np.dot(np.dot(S2m_p, V_).transpose(), Jvi_), np.dot(S2m_c, V_))
    graph = nx.from_numpy_array(mass_flow_graph, create_using=nx.DiGraph)
    pagerank = pd.DataFrame(nx.pagerank(graph, alpha=0.90), index=["pagerank"])

    return pagerank
