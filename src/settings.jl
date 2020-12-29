#=
settings:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-12-29
Settings for different structs
=#

""" Returns default BlackBoxFunction settings for approximation."""
function bbf_defaults()
    Dict(:threshold_accuracy => 0.95,      # Minimum tree accuracy
         :threshold_feasibility => 0.15,   # Minimum feasibility ratio
         :reloaded => false)
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
        gm.settings[key] = val
    else
        throw(OCTException("Parameter of key " * string(key) *" is invalid.")
    end
end