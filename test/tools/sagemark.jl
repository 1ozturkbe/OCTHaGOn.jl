#=
test_tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Tests tools.
=#

using PyCall

function alphac_to_expr(model, alpha, c, lse::Bool = false)
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

function alphac_to_varbound!(model, alpha, c, lse::Bool = false)
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

function sagemark_to_GlobalModel(idx, lse::Bool = false)
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
    gm = GlobalModel(model = model, name = "sagemark" * string(idx))
    obj_fn, objvars = alphac_to_expr(gm.model, f.alpha, f.c, lse)
    add_nonlinear_constraint(gm, obj_fn, vars = objvars, dependent_var = obj)
    # Adding the rest
    for i = 1:length(greaters)
        constrexpr, constrvars = alphac_to_expr(gm.model, greaters[i].alpha, greaters[i].c, lse)
#         if sum(float(greaters[i].c .>= zeros(length(greaters[i].c)))) == 1
#         TODO: add tag for convexity
        if length(constrvars) > 1
            add_nonlinear_constraint(gm, constrexpr, vars = constrvars)
        else
            alpha = greaters[i].alpha
            c = greaters[i].c
            alphac_to_varbound!(gm.model, alpha, c, lse)
        end
    end
    for i = 1:length(equals)
        constrexpr, constrvars = alphac_to_expr(gm.model, equals[i].alpha, equals[i].c, lse)
        add_nonlinear_constraint(gm, constrexpr, vars = constrvars, equality=true)
    end
    set_optimizer(gm, CPLEX_SILENT)
    return gm
end