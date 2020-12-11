#=
gams:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-12-10
=#

"""Returns all constants from GAMS Model. """
function constants(gams::Dict{String, Any})
    consts = Dict{String, Any}(); #
    for p in ("parameters", "tables")
        if haskey(gams, p)
            for (k, v) in gams[p]
                v = v isa Ref ? v[] : v
                consts[GAMSFiles.getname(k)] = v;
            end
        end
    end
    return consts
end

""" Moves GAMS constants into the global scope.
    NOTE: This is only for debugging purposes, could cause bugs.
"""
function constants_to_global(gams::Dict{String, Any})
    consts = constants(gams)
    for (key, value) in consts
        eval(Meta.parse("$(key) = $(value)"))
    end
end