#=
plot:
- Julia version: 
- Author: Berk
- Date: 2020-07-28
=#

function plot(bbf::BlackBoxFunction)
    """ Plots a BlackBoxFunction, which is either a 2D scatter plot or a 2D Gaussian Process. """
    if !isnothing(bbf.gp)
        if bbf.gp.dim > 1
            throw(OCTException("plot(BlackBoxFunction) only works in 2D."))
        end
        Plots.plot(bbf.gp, legend=false, fmt=:png)
    else
        if size(bbf.X, 2) > 1
            throw(OCTException("plot(BlackBoxFunction) only works in 2D."))
        end
        scatter(Matrix(bbf.X), bbf.Y, legend=false, fmt=:png)
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