#=
plot:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-12-10
=#

""" Plots the feasible and infeasible samples of BBF in two axes. """
function plot_2d(bbf::Union{BlackBoxFunction, DataConstraint}; axis1 = 1, axis2 = 2)
    feas_idxs = findall(x -> x .>= 0, bbf.Y)
    infeas_idxs = findall(x -> x .< 0, bbf.Y)
    scatter(Matrix(bbf.X)[feas_idxs,axis1], Matrix(bbf.X)[feas_idxs, axis2],
            legend = false, color = :green,  fmt = :png)
    scatter!(Matrix(bbf.X)[infeas_idxs, axis1], Matrix(bbf.X)[infeas_idxs, axis2],
            title = "Feasible vs. infeasible samples",
    legend = false, color = :red,  fmt = :png)
end

""" Plots whether or not model predictions are correct, in 2D. """
function plot_2d_predictions(bbf::Union{BlackBoxFunction, DataConstraint}; axis1 = 1, axis2 = 2)
    Y = IAI.predict(bbf.learners[end], bbf.X)
    trues = [i for i=1:length(Y) if Y[i] == (bbf.Y[i] >= 0)];
    falses = [i for i=1:length(Y) if Y[i] != (bbf.Y[i] >= 0)];
    scatter(Matrix(bbf.X)[trues, axis1], Matrix(bbf.X)[trues, axis2], legend = false,
            color = :purple,  fmt = :png)
    scatter!(Matrix(bbf.X)[falses, axis1], Matrix(bbf.X)[falses, axis2], legend = false,
             title = "Correct vs. incorrect predictions",
             color = :yellow,  fmt = :png)
end