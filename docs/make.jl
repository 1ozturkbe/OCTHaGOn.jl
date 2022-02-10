using Documenter
using DocStringExtensions

makedocs(sitename = "OCTHaGOn",
        format = Documenter.HTML(
            prettyurls = get(ENV, "CI", nothing) == "true"
        )
        )