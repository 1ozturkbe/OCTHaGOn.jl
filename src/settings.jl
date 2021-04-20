""" Returns default BlackBoxRegressor settings for approximation."""
function bbr_defaults(n_vars::Int64 = 10; sample_coeff = 200)
    Dict(:n_samples => Int(ceil(sample_coeff*sqrt(n_vars))), # (0 if no sampling fn)
        :gradients => true,                                 # TODO: add option whether or not to use gradients
        :reloaded => false)                                 # Whether learners are reloaded            
end

""" Returns default BlackBoxClassifier settings for approximation."""
function bbc_defaults(n_vars::Int64 = 10; sample_coeff = 200)
    Dict(:threshold_accuracy => 0.95,                       # Minimum tree accuracy
        :threshold_feasibility => 0.15,                     # Minimum feasibility ratio
        :ignore_feasibility => false,                       # Whether we should ignore feasibility checks
        :ignore_accuracy => false,                          # Whether we should ignore accuracy checks 
        :n_samples => Int(ceil(sample_coeff*sqrt(n_vars))), # (0 if no sampling fn)
        :gradients => true,
        :reloaded => false,                                 # Whether learners are reloaded  
        :linking => false,
        :linked_vars => [])
end

"""
Returns default GlobalModel settings for approximation and optimization.
"""
function gm_defaults()
    Dict(:ignore_feasibility => false,
         :ignore_accuracy => false,
         :lh_iterations => 5,
         :sample_coeff => 200,
         :sample_density => 1e-5,
         :abstol => 1e-3,
         :tighttol => 1e-5,
         :reltol => 1e-3,
         :convex_constrs => false)
end

""" Sets parameters within Dict. """
function set_param(gm::Dict, key::Symbol, val, checks = true)
    if haskey(gm, key) && val isa typeof(gm[key])
        gm[key] = val
        return
    elseif checks
        throw(OCTException("Parameter with key " * string(key) * " is invalid."))
    else 
        return
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