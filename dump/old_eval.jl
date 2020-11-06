#=
old_eval:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-11-05
=#

function evaluate(constraint::JuMP.ConstraintRef, data::Union{Dict, DataFrame})
    constr_obj = constraint_object(constraint)
    rhs_const = get_constant(constr_obj.set)
    clean_data = data_to_DataFrame(data);
    vals = [JuMP.value(constr_obj.func, i -> get(clean_data, string(i), Inf)[j]) for j=1:size(clean_data,1)]
    if any(isinf.(vals))
        throw(OCTException(string("Constraint ", constraint, " returned an infinite value.
                                   Make sure you have declared all bbf.vars!")))
    end
    if isnothing(rhs_const)
        vals = [-1*distance_to_set(vals[i], constr_obj.set) for i=1:length(vals)]
    else
        coeff=1
        if constr_obj.set isa MOI.LessThan
            coeff=-1
        end
        vals = coeff.*(vals .- rhs_const)
    end
    length(vals) == 1 && return vals[1]
    return vals
end

function evaluate(constraint::JuMP.ConstraintRef, data::Union{Dict, DataFrame},
                           model::JuMP.Model)
   return evaluate(constraint, data)
end

"""
    evaluate(constraint::Expr, data::Union{Dict, DataFrame}, model::JuMP.Model)

    Evaluates constraint violation on data in an expression.
    Returns Array of values.

    Note that the keys of the Dict have to be uniform.
"""
function evaluate(constraint::Function, data::Union{Dict, DataFrame}, vars::Array)
    args = []
    clean_data = data_to_DataFrame(data)
    arrays = [[j=1:length()] for i=1:size(clean_data,1)]
    for var_element in vars

    n_vals, n_vars = size(clean_data)
    for (outer, vars) in outer_var_mapping
        if vars isa JuMP.VariableRef
            new_data[outer] = clean_data[!, string(vars)]
        elseif vars isa Array{JuMP.VariableRef}
            new_data[outer] = Inf*ones(size(clean_data,1), length(vars))
            for i=1:length(vars)
                if haskey(clean_data, string(vars[i]))
                    new_data[outer][:,i] = clean_data[!, string(vars[i])]
                end
            end
        else
            throw(ArgumentError(string(constraint, "had an error in evaluation.")))
        end
    end
    new_data = chop_dict(new_data) # makes sure to "scalarize" scalar variable inputs.
    if n_vals == 1
        return substitute_expr(constraint, new_data[1])
    else
        return [substitute_expr(constraint, new_data[i]) for i=1:length(new_data)]
    end
end

function evaluate(constraint::Expr, data::Union{Dict, DataFrame}, vars::Array)
    f = eval(constraint)
    @assert f isa Function
    evaluate(f, data, vars)
end