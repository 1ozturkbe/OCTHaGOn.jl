""" Returns default BlackBoxFunction settings for approximation."""
function bbf_defaults()
    Dict(:threshold_accuracy => 0.95,      # Minimum tree accuracy
         :threshold_feasibility => 0.15,   # Minimum feasibility ratio
         :n_samples => 200,                # Maximum number of samples at each step
         :regression => false,             # Whether trees should be regression trees (not implemented)
         :reloaded => false)               # Whether learners are reloaded
end

"""
Returns default GlobalModel settings for approximation and optimization.
"""
function gm_defaults()
    Dict(:ignore_feasibility => false,
         :ignore_accuracy => false,
         :linear => true,
         :convex => false)
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