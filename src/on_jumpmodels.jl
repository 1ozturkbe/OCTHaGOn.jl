#=
on_jumpmodels:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-09-25
=#


function get_nonlinear_constraints(model::JuMP.Model)
    constr_types = JuMP.list_of_constraint_types(model)
    nl_constraints = all_constraints(model::Model, function_type, set_type)
    return nl_constraints
end

func

function get_variables(constraint::JuMP.ConstraintRef)
    if constraint isa JuMP.NonlinearConstraintRef
    else
       return constraint.terms
    end
end

function get_bounds(model::JuMP.Model)
    all_vars = JuMP.all_variables(model)
    bounds = Dict(string(var) => [-Inf, Inf] for var in all_vars);
    for var in all_vars
        if JuMP.has_lower_bound(var)
            bounds[string(var)][1] = JuMP.lower_bound(var);
        end
        if JuMP.has_upper_bound(var)
            bounds[string(var)][2] = JuMP.upper_bound(var);
        end
    end
    return bounds
end

function update_lower_bound(var::JuMP.VariableRef, val::Real)
    """ Updates variable lower bounds consistently. """
    if JuMP.has_lower_bound(d)
        if val >= lower_bound(d)
            JuMP.set_lower_bound(d, val)
        end
        return
    else
        JuMP.set_lower_bound(d, val)
        return
    end
end

function update_upper_bound(var::JuMP.VariableRef, val::Real)
    """ Updates variable upper bounds consistently. """
    if JuMP.has_upper_bound(d)
        if val <= upper_bound(d)
            JuMP.set_upper_bound(d, val)
        end
        return
    else
        JuMP.set_upper_bound(d, val)
        return
    end
end


# JuMP.register for new nonlinear functions