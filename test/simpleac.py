# sagemarks = pyimport("sagebenchmarks.literature.solved");

# pushfirst!(PyVector(pyimport("sys")."path"), "")
# simpleac = pyimport("data/SimPleAC_mission.py")
# for (idx, val) in bbl_idxs
#     signomials, solver, run_fn = sagemarks.get_example(idx);
#     f, greaters, equals = signomials;
#     check_cvx_obj(f.alpha, f.c, lse) && push!(actual_cvx[idx], 1)
#     for i = 1:length(greaters)
#         # if to prune the bounds
#         if !(sum(greaters[i].alpha .!= 0) == 1 && sum(greaters[i].alpha) == 1) && i+1 in val
#             check_cvx_con(greaters[i].alpha, greaters[i].c, lse) && 
#             push!(actual_cvx[idx], i+1)
#         end
#     end
# end
# actual_cvx = Dict(key => value for (key, value) in actual_cvx if !isempty(value))


import sageopt as so
import numpy as np
from sageopt.interop.gpkit import gpkit_model_to_sageopt_model
from gpkit import Variable, Model, SignomialsEnabled
from gpkit.nomials import SingleSignomialEquality

#
# Build GPKit model
#
x = Variable('x')
y = Variable('y')
with SignomialsEnabled():
    constraints = [0.2 <= x, x <= 0.95, SingleSignomialEquality(x + y, 1)]
gpk_mod = Model(x*y, constraints)

from sageopt.interop.gpkit import *

def gpkit_hmap_to_sageopt_sig(curhmap, vkmap):
    n_vks = len(vkmap)
    temp_sig_dict = dict()
    for expinfo, coeff in curhmap.items():
        tup = n_vks * [0]
        for vk, expval in expinfo.items():
            tup[vkmap[vk]] = expval
        temp_sig_dict[tuple(tup)] = coeff
    s = Signomial.from_dict(temp_sig_dict)
    return s


def gp_con_hmap(con, subs):
    exprlist = con.as_hmapslt1({})
    if len(exprlist) == 0:
        return None
    else:
        hmap = exprlist[0].sub(subs, exprlist[0].keys())
        return hmap


def sp_eq_con_hmap(con, subs):
    expr = con.right - con.left
    hmap = expr.sub(subs).hmap
    return hmap


def sp_ineq_con_hmap(con, subs):
    gtzero_rep = (con.right - con.left) * (1. - 2 * (con.oper == '>='))
    hmap = gtzero_rep.sub(subs).hmap
    return hmap

subs = gpk_mod.substitutions
constraints = [con for con in gpk_mod.flat()]
varkeys = sorted([vk for vk in gpk_mod.varkeys if vk not in subs], key=lambda vk: vk.latex())
vkmap = {vk: i for (i, vk) in enumerate(varkeys)}
# construct sageopt Signomial objects for each GPKit constraint
gp_eqs, gp_gts, sp_eqs, sp_gts = [], [], [], []


for i, constraint in enumerate(constraints):
    if isinstance(constraint, MonomialEquality):
        hmap = gp_con_hmap(constraint, subs)
        if hmap is not None:
            cursig = 1 - gpkit_hmap_to_sageopt_sig(hmap, vkmap)
            cursig.metadata['GPKit constraint index'] = i
            gp_eqs.append(cursig)
    elif isinstance(constraint, PosynomialInequality):
        hmap = gp_con_hmap(constraint, subs)
        if hmap is not None:
            cursig = 1 - gpkit_hmap_to_sageopt_sig(hmap, vkmap)
            cursig.metadata['GPKit constraint index'] = i
            gp_gts.append(cursig)
    elif isinstance(constraint, SignomialInequality):
        # ^ incidentally, these can also be equality constraints
        with SignomialsEnabled():
            if isinstance(constraint, SingleSignomialEquality):
                hmap = sp_eq_con_hmap(constraint, subs)
                cursig = gpkit_hmap_to_sageopt_sig(hmap, vkmap)
                cursig.metadata['GPKit constraint index'] = i
                sp_eqs.append(cursig)
            else:
                hmap = sp_ineq_con_hmap(constraint, subs)
                cursig = gpkit_hmap_to_sageopt_sig(hmap, vkmap)
                cursig.metadata['GPKit constraint index'] = i
                sp_gts.append(cursig)
# Build a sageopt Signomial from the GPKit objective.
objective_hmap = gpk_mod.cost.hmap.sub(subs, gpk_mod.varkeys)
f = gpkit_hmap_to_sageopt_sig(objective_hmap, vkmap)
# Somehow aggregate all sageopt problem data. Might make a class for this later.
so_mod = {
    'f': f,
    'gp_eqs': gp_eqs,
    'sp_eqs': sp_eqs,
    'gp_gts': gp_gts,
    'sp_gts': sp_gts,
    'vkmap': vkmap
}


# # Problematic code here
# som = gpkit_model_to_sageopt_model(gpk_mod)  # a dict
# sp_eqs, gp_gts = som['sp_eqs'], som['gp_gts']
# f = som['f']
# X = so.infer_domain(f, gp_gts, [])
# prob = so.sig_constrained_relaxation(f, gp_gts, sp_eqs, X,  p=1)
# #
# #   Solve and check solution
# #
# prob.solve(solver='ECOS', verbose=False)
# soln = so.sig_solrec(prob)[0]
# geo_soln = np.exp(soln)
# vkmap = som['vkmap']
# x_val = geo_soln[vkmap[x.key]]
# y_val = geo_soln[vkmap[y.key]]