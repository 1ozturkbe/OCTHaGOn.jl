#=
tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Adds tools to mainly import optimization problems from other sources.
=#

function alphac_to_NLexpr(model, alpha, c; lse=false)
    """ Turns exponential to JuMP.NonlinearExpression. """
    n_terms, n_vars = size(alpha)
    idxs = unique([i[2] for i in findall(i->i != 0, alpha)]);
    vars = JuMP.all_variables(model)[idxs]
    alpha = alpha[:,idxs];
    if lse
        return @NLexpression(model, sum(c[i]*exp(sum(alpha[i,j]*vars[j] for j=1:length(vars))) for i=1:n_terms)), vars
    else
        return @NLexpression(model, sum(c[i]*prod(vars[j]^alpha[i,j] for j=1:length(vars)) for i=1:n_terms)), vars
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
        md:: a function_model
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
    obj_fn, objvars = alphac_to_NLexpr(model, f.alpha, f.c, lse=lse)
    push!(objvars, obj)
    gm = GlobalModel(model = model)
    if lse
        add_nonlinear_constraint(gm, @NLexpression(gm.model, exp(obj) - obj_fn), vars = objvars)
    else
        add_nonlinear_constraint(gm, @NLexpression(gm.model, obj - obj_fn), vars = objvars)
    end
    # Adding the rest
    for i = 1:length(greaters)
        constrexpr, constrvars = alphac_to_NLexpr(gm.model, greaters[i].alpha, greaters[i].c, lse=lse)
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
        constrexpr, constrvars = alphac_to_NLexpr(model, equals[i].alpha, equals[i].c, lse=lse)
        add_nonlinear_constraint(gm, constrexpr, vars = constrvars, equality=true)
    end
    return gm
end

