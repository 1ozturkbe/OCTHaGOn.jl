#=
tools:
- Julia version: 
- Author: Berk
- Date: 2020-06-17
Adds tools to mainly import optimization problems from other sources.
=#

function alphac_to_NLexpr(model, alpha, c; lse=false)
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

function alphac_to_varbound(model, alpha, c; lse=false)

end

function sagemark_to_ModelData(idx; lse=false)
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
    # Assigning objective
    obj, objvars = alphac_to_NLexpr(model, f.alpha, f.c, lse=lse)
    @NLobjective(model, Min, obj)
    # Adding the rest
    for i = 1:length(greaters)
        constrexpr, constrvars = alphac_to_NLexpr(model, greaters[i].alpha, greaters[i].c, lse=lse)
#         if sum(float(greaters[i].c .>= zeros(length(greaters[i].c)))) == 1
#         TODO: add tag for convexity
        if length(constrvars) > 1
            @NLconstraint(model, constrexpr >= 0)
        else
            alpha = greaters[i].alpha
            c = greaters[i].c
            idx = unique([j for j in findall(j->j != 0, alpha)])[1]; #idx[1] is monomial index,
            var = x[idx[2]];                                                     #idx[2] is variable index.
            val = -((sum(c)-c[idx[1]]) / c[idx[1]])^(1/alpha[idx]);
            lse && (val = log(val))
            if c[idx[1]] <= 0
                bound!(model,  Dict(x[idx[2]] => [-Inf, val]));
            else
                bound!(model,  Dict(x[idx[2]] => [val, Inf]));
            end
        end
    end
    for i = 1:length(equals)
        constrexpr, constrvars = alphac_to_NLexpr(model, equals[i].alpha, equals[i].c, lse=lse)
        @NLconstraint(model, constrexpr == 0)
    end
    return model
end

# function MOF_to_ModelData(mof_model)
# TODO: Work in progress...
#     constraints = MathOptInterface.get(mof_model, MathOptInterface.ListOfConstraints())
#     constraint_indices = MathOptInterface.get(mof_model, MathOptInterface.ListOfConstraintIndices)
#     variable_attributes_set = MathOptInterface.get(mof_model, MathOptInterface.ListOfVariableAttributesSet())
#     #TODO: CONTINUE HERE
#     obj = model.objective.constant*model.objective.terms
#     lbs = model.lower_bounds;
#     ubs = model.upper_bounds;
#     ex = function_model(string("example", idx),
#                         obj, obj_idxs, ineqs, ineq_idxs,
#                         eqs, eq_idxs, lbs, ubs, lse);
#     return ex
# end