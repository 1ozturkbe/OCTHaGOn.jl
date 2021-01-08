""" Returns default BlackBoxRegressor settings for approximation."""
function bbr_defaults(n_vars::Int64 = 10; sample_coeff = 200)
    Dict(:n_samples => Int(ceil(sample_coeff*sqrt(n_vars))), # (0 if no sampling fn)
         :reloaded => false)                                 # Whether learners are reloaded
end

""" Returns default BlackBoxClassifier settings for approximation."""
function bbc_defaults(n_vars::Int64 = 10; sample_coeff = 200)
    Dict(:threshold_accuracy => 0.95,                       # Minimum tree accuracy
        :threshold_feasibility => 0.15,                     # Minimum feasibility ratio
        :n_samples => Int(ceil(sample_coeff*sqrt(n_vars))), # (0 if no sampling fn)
        :reloaded => false)                                 # Whether learners are reloaded  
end

"""
Returns default GlobalModel settings for approximation and optimization.
"""
function gm_defaults()
    Dict(:ignore_feasibility => false,
         :ignore_accuracy => false,
         :linear => true,
         :convex => false,
         :lh_iterations => 5,
         :sample_coeff => 200)
end

""" Sets parameters within Dict. """
function set_param(gm::Dict, key::Symbol, val)
    if haskey(gm, key) && val isa typeof(gm[key])
        gm[key] = val
    else
        throw(OCTException("Parameter with key " * string(key) * " is invalid."))
    end
end

""" gets parameters within Dict. """
function get_param(gm::Dict, key::Symbol)
    if haskey(gm, key)
        return gm[key]
    else
        throw(OCTException("Parameter with key " * string(key) * " is invalid."))
    end
end