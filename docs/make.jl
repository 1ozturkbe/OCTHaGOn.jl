using Pkg

include("../test/load.jl")

Pkg.activate("docs/.")

using Documenter
using DocStringExtensions

makedocs(sitename = "OCTHaGOn",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [OCTHaGOn],
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Basic usage" => "basic.md",
        "API reference" => [
            "api/constraints.md",
            "api/sampling.md",
            "api/iai_wrappers.md",
            "api/datastructures.md",
            "api/helpers.md"
        ]
    ]
)

# deploydocs(deps = nothing, make = nothing, 
#   repo = "github.com/1ozturkbe/OCTHaGOn.jl.git",
#   target = "build",
#   push_preview = true)