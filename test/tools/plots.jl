# Plotting data along two X-axes, in original space
using Plots

""" Plots the feasible and infeasible samples of BlackBoxClassifier in two axes. """
function plot_2d(bbc::BlackBoxClassifier; axes = [1, 2])
    feas_idxs = findall(x -> x .>= 0, bbc.Y)
    infeas_idxs = findall(x -> x .< 0, bbc.Y)
    scatter(Matrix(bbc.X)[feas_idxs, axes[1]], Matrix(bbc.X)[feas_idxs, axes[2]],
            legend = false, color = :green,  fmt = :png)
    scatter!(Matrix(bbc.X)[infeas_idxs, axes[1]], Matrix(bbc.X)[infeas_idxs, axes[2]],
            title = "Feasible vs. infeasible samples",
    legend = false, color = :red,  fmt = :png)
end

""" Plots whether or not BlackBoxClassifier predictions are correct, in 2D. """
function plot_2d_predictions(bbc::BlackBoxClassifier; axes = [1, 2])
    Y = IAI.predict(bbc.learners[end], bbc.X)
    trues = [i for i=1:length(Y) if Y[i] == (bbc.Y[i] >= 0)];
    falses = [i for i=1:length(Y) if Y[i] != (bbc.Y[i] >= 0)];
    scatter(Matrix(bbc.X)[trues, axes[1]], Matrix(bbc.X)[trues, axes[2]], legend = false,
            color = :purple,  fmt = :png)
    scatter!(Matrix(bbc.X)[falses, axes[1]], Matrix(bbc.X)[falses, axes[2]], legend = false,
             title = "Correct vs. incorrect predictions",
             color = :yellow,  fmt = :png)
end

""" Plots the data in BlackBoxRegressor."""
function plot_2d(bbr::BlackBoxRegressor; axes = [1, 2])
    plt3d = Plots.plot(bbr.X[:, axes[1]], bbr.X[:, axes[2]], bbr.Y,
    seriestype=:surface, markersize = 2, camera=(20,30))
    xlabel!(string(bbr.vars[axes[1]]))                   
    ylabel!(string(bbr.vars[axes[2]]))                   
    title!("Data")
    display(plt3d)  
end

""" Plots whether or not BlackBoxRegressor predictions are correct, in 2D. """
function plot_2d_predictions(bbr::BlackBoxRegressor; axes = [1, 2])  
    plt3d2 = Plots.plot(bbr.X[:, axes[1]], bbr.X[:, axes[2]], bbr.predictions,
    seriestype=:surface, markersize = 2, camera=(20,30))     
    xlabel!(string(bbr.vars[axes[1]]))                   
    ylabel!(string(bbr.vars[axes[2]]))                   
    title!("Predictions")
    display(plt3d2)
end

""" Plots the accuracies of different BlackBoxLearners. """
function plot_accuracies(bbls::Array{BlackBoxClassifier})
        bar(1:length(bbls), [bbl.accuracies[end] for bbl in bbls], xlabel="Constraint number",
                        xticks=1:length(ineqs), title="Constraint accuracies", legend=false)
end