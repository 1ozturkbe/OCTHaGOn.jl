#=
test_tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Tests tools.
=#

using PyCall
using Ipopt

function alphac_to_expr(model, alpha, c; lse=false)
    """ Turns exponential to JuMP.NonlinearExpression. """
    n_terms, n_vars = size(alpha)
    idxs = unique([i[2] for i in findall(i->i != 0, alpha)]);
    vars = JuMP.all_variables(model)[idxs];
    if lse
        return :(x -> sum($c[i]*exp(sum($alpha[i,j]*x[j] for j in $idxs)) for i=1:$n_terms)), vars
    else
        return :(x -> sum($c[i]*prod(x[j]^$alpha[i,j] for j in $idxs) for i=1:$n_terms)), vars
    end
end


function alphac_to_objexpr(model, alpha, c; lse=false)
    n_terms, n_vars = size(alpha)
    idxs = unique([i[2] for i in findall(i->i != 0, alpha)]);
    vars = flat([JuMP.all_variables(model)[idxs], model[:obj]]);
    if lse
        return :((x, obj) -> exp(obj) - sum($c[i]*exp(sum($alpha[i,j]*x[j] for j in $idxs)) for i=1:$n_terms)), vars
    else
        return :((x, obj) -> obj - sum($c[i]*prod(x[j]^$alpha[i,j] for j in $idxs) for i=1:$n_terms)), vars
    end
end

function alphac_to_varbound!(model, alpha, c; lse=false)
    """ Turns univariate exponential to JuMP variable bounds. """
    idx = unique([j for j in findall(j->j != 0, alpha)])[1]; #idx[1] is monomial index,
    var = JuMP.all_variables(model)[idx[2]];                 #idx[2] is variable index.
    val = -((sum(c)-c[idx[1]]) / c[idx[1]])^(1/alpha[idx]);
    lse && (val = log(val))
    if c[idx[1]] <= 0
        set_upper_bound(var, val);
    else
        set_lower_bound(var, val);
    end
end

function sagemark_to_GlobalModel(idx; lse=false)
    """
    Imports sagebenchmarks example from literature.solved and
    returns as a function_model.
    Arguments:
        idx:: number of benchmark
    Returns:
        gm:: a function_model
    """
    sagemarks = pyimport("sagebenchmarks.literature.solved");
    signomials, solver, run_fn = sagemarks.get_example(idx);
    f, greaters, equals = signomials;
    n_vars = size(f.alpha,2);
    model = JuMP.Model()
    @variable(model, x[1:n_vars])
    @variable(model, obj)
    @objective(model, Min, obj)
    # Assigning objective
    gm = GlobalModel(model = model)
    obj_fn, objvars = alphac_to_objexpr(gm.model, f.alpha, f.c, lse=lse)
    add_nonlinear_constraint(gm, obj_fn, vars = objvars)
    # Adding the rest
    for i = 1:length(greaters)
        constrexpr, constrvars = alphac_to_expr(gm.model, greaters[i].alpha, greaters[i].c, lse=lse)
#         if sum(float(greaters[i].c .>= zeros(length(greaters[i].c)))) == 1
#         TODO: add tag for convexity
        if length(constrvars) > 1
            add_nonlinear_constraint(gm, constrexpr, vars = constrvars)
        else
            alpha = greaters[i].alpha
            c = greaters[i].c
            alphac_to_varbound!(gm.model, alpha, c, lse=lse)
        end
    end
    for i = 1:length(equals)
        constrexpr, constrvars = alphac_to_expr(gm.model, equals[i].alpha, equals[i].c, lse=lse)
        add_nonlinear_constraint(gm, constrexpr, vars = constrvars, equality=true)
    end
    return gm
end

function test_sagemark_to_GlobalModel()
    """ Makes sure all sage benchmarks import properly.
        For now, just doing first 25, since polynomial
        examples are not in R+. """
    @info("Testing sagemark imports.")
    idxs = 1:25;
    # max_min = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx in idxs
        ex = sagemark_to_GlobalModel(idx);
    end
    idx = 1;
    gm = sagemark_to_GlobalModel(idx; lse=false);
    gm_lse = sagemark_to_GlobalModel(idx; lse=true);
    classify_constraints.([gm, gm_lse])
    return true
end

# Importing sagebenchmark to ModelData and checking it
@test test_sagemark_to_GlobalModel()
gm = sagemark_to_GlobalModel(1, lse=false);
gm_lse = sagemark_to_GlobalModel(1, lse=true);
# set_optimizer(gm, Ipopt.Optimizer)
# nonlinearize!(gm)
# optimize!(gm)
# vals = getvalue.(gm.vars)

bounds = Dict(all_variables(gm) .=> [[0.1, 1], [5., 10.], [8., 15.], [0.01, 1.], [-Inf, Inf]])
@test all(get_bounds(gm)[var] ≈ bounds[var] for var in all_variables(gm))
inp = Dict(all_variables(gm) .=> [1,1.9,3,3.9, 10.]);
log_inp = Dict(vk => log(val) for (vk, val) in inp);

@test gm.vars == all_variables(gm);
@test gm_lse.bbfs[1](log_inp)[1] ≈ gm.bbfs[1](inp)[1] ≈ [inp[gm.vars[5]] - inp[gm.vars[3]] ^ 0.8 * inp[gm.vars[4]] ^ 1.2][1]

# Checking OCTException for sampling unbounded model
@test_throws OCTException sample_and_eval!(gm.bbfs[1], n_samples=200)

# Actually trying to optimize...
gm = sagemark_to_GlobalModel(3; lse=false);
set_optimizer(gm, Gurobi.Optimizer)
old_bounds = get_bounds(gm)
find_bounds!(gm, all_bounds=true)
@test old_bounds == get_bounds(gm)
bound!(gm, Dict(gm.vars[end] => [-300, 0]))
sample_and_eval!(gm, n_samples = 500)
sample_and_eval!(gm, n_samples = 500)

learn_constraint!(gm)
println("Approximation accuracies: ", accuracy(gm))

# Solving of model
@test_throws OCTException globalsolve(gm) # inaccuracy check in globalsolve.
gm.settings[:ignore_accuracy] = true
globalsolve(gm);
println("X values: ", solution(gm))
println("Optimal X: ", vcat(exp.([5.01063529, 3.40119660, -0.48450710]), [-147-2/3]))

# Testing constraint addition and removal
clear_tree_constraints!(gm) # Clears all BBF constraints
@test all([!is_valid(gm.model, constraint) for constraint in gm.bbfs[2].mi_constraints])
add_tree_constraints!(gm, [gm.bbfs[2]])
@test all([is_valid(gm.model, constraint) for constraint in gm.bbfs[2].mi_constraints])
add_tree_constraints!(gm);
clear_tree_constraints!(gm, [gm.bbfs[1]])
@test !any(is_valid(gm.model, constraint) for constraint in gm.bbfs[1].mi_constraints)
clear_tree_constraints!(gm) # Finds and clears the one remaining BBF constraint.
@test all([!is_valid(gm.model, constraint) for constraint in gm.bbfs[1].mi_constraints])
@test all([!is_valid(gm.model, var) for var in gm.bbfs[1].leaf_variables])