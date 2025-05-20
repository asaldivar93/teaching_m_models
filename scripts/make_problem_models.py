from __future__ import annotations


import cobra


folder = "models/exercises/gapfilling/"
model = cobra.io.read_sbml_model("models/external/mfum_ch4.xml")
model.solver="glpk"
cobra.io.write_sbml_model(model, "models/external/mfum_ch4.xml")

print(model.slim_optimize())

#gapfilled with genes
with model as prob:
    prob.remove_reactions(["GLCS1"])
    cobra.io.write_sbml_model(prob, folder + "gf_ex1.xml")

# gapfilled without genes
with model as prob:
    prob.remove_reactions(["HSST", "SHSL1"])
    cobra.io.write_sbml_model(prob, folder + "gf_ex2.xml")


# Dead end metabolite with Demand reaction
with model as prob:
    prob.remove_reactions(["DM_amob_c"])
    cobra.io.write_sbml_model(prob, folder + "gf_ex3.xml")
