struct function_model
    name::String            # Example name
    obj::Function           # Objective f'n
    obj_idxs::Array         # Objective indices
    constr::Array{Function} # Constraints
    constr_idxs::Array      # Constraint indices
    lbs::Array              # Lower bounds
    ubs::Array              # Upper bounds
end