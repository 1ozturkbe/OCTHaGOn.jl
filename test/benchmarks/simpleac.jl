pushfirst!(PyVector(pyimport("sys")."path"), "")
models = pyimport("robust.testing.models")
interop = pyimport("sageopt.interop.gpkit")

simpleac = models.simple_ac()
sm = interop.gpkit_model_to_sageopt_model(models.simple_ac())

m = length(simpleac)

# Dissecting all of the constraints

function gpkit_to_sageopt_GlobalModel(m)
    py"""
    from sageopt.interop.gpkit import gpkit_model_to_sageopt_model
    def convert_the_models(m):
        return gpkit_model_to_sageopt_model
    """
    return py"convert_the_models(m)"
end


# function gpkit_to_GlobalModel(m, lse::Bool = False)
#     """
#     Imports sagebenchmarks example from literature.solved and
#     returns as a function_model.
#     Arguments:
#         m:: A Gpkit Model
#     Returns:
#         sm:: Corresponding sageopt model, with a varkey map vkmap
#     """

#     model = JuMP.Model()
#     @variable(model, x[1:n_vars])
#     @variable(model, obj)
#     @objective(model, Min, obj)
#     # Assigning objective
#     gm = GlobalModel(model = model, name = "sagemark" * string(idx))
#     obj_fn, objvars = alphac_to_expr(gm.model, f.alpha, f.c, lse)
#     add_nonlinear_constraint(gm, obj_fn, vars = objvars, dependent_var = obj)
#     # Adding the rest
#     for i = 1:length(greaters)
#         constrexpr, constrvars = alphac_to_expr(gm.model, greaters[i].alpha, greaters[i].c, lse)
# #         if sum(float(greaters[i].c .>= zeros(length(greaters[i].c)))) == 1
# #         TODO: add tag for convexity
#         if length(constrvars) > 1
#             add_nonlinear_constraint(gm, constrexpr, vars = constrvars)
#         else
#             alpha = greaters[i].alpha
#             c = greaters[i].c
#             alphac_to_varbound!(gm.model, alpha, c, lse)
#         end
#     end
#     for i = 1:length(equals)
#         constrexpr, constrvars = alphac_to_expr(gm.model, equals[i].alpha, equals[i].c, lse)
#         add_nonlinear_constraint(gm, constrexpr, vars = constrvars, equality=true)
#     end
#     set_optimizer(gm, CPLEX_SILENT)
#     return gm
# end

# Python version

import robust.testing.models as models
m = models.simple_ac()