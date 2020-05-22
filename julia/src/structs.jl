struct function_model
    name::String            # Example name
    obj::Function           # Objective f'n
    obj_idxs::Array         # Objective indices
    ineqs::Array{Function}  # Inequality (>= 0) constraints
    ineq_idxs::Array        # Constraint indices
    eqs::Array{Function}    # Equality (== 0) constraints
    eq_idxs::Array          # Equality indices
    lbs::Array              # Lower bounds
    ubs::Array              # Upper bounds
    lse::Bool
end
