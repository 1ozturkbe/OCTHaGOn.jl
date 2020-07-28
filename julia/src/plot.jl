#=
plot:
- Julia version: 
- Author: Berk
- Date: 2020-07-28
=#

include("black_box_function.jl")

function plot(bbf::BlackBoxFn)
    """ Plots a BlackBoxFn, which is either a 2D scatter plot or a 2D Gaussian Process. """
    if !isnothing(bbf.gp)
        plot(gp, xlabel="x", ylabel="y", title="Gaussian process", legend=false, fmt=:png)
    else
        scatter(Matrix(bbf.samples), bbf.values, xlabel="x", ylabel="y",
             title="Gaussian process", legend=false, fmt=:png)
    end
end

function plot(tree::IAI.GridSearch)
    """ Shows grid in browser. """
    try
        IAI.show_in_browser(tree.lnr)
    catch err
        if isa(err, UndefRefError)
        @warn("Certain trees are untrained.")
        else
            rethrow(err)
        end
    end
end