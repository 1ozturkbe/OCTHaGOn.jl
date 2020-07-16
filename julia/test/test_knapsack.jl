#=
test_knapsack:
- Julia version: 
- Author: Berk
- Date: 2020-07-01
=#

using Test
using DataFrames
using Distributions
using ExperimentalDesign
using Gurobi
using JuMP
using Random
Random.seed!(1);
include("../src/OptimalConstraintTree.jl");
global OCT = OptimalConstraintTree;


function knapsack(c, a, b; name="knapsack", lbs = zeros(length(c)), ubs = ones(length(c)),
                  int_idxs = [i for i=1:length(c)])
    """ Creates a knapsack ModelData problem"""
    return OCT.ModelData(c=-c, name=name,
                      ineqs_A = [-a], ineqs_b = [b], lbs=lbs, ubs=ubs, int_idxs=int_idxs)
end

# function knapsack_jump(c,a,b)
#     m = Model(solver=GurobiSolver())
#     @variable(m, x[1:length(c)], Bin)
#     @constraint(m, sum(a.*x) <= b)
#     @objective(m, Max, sum(c.*x))
#     return m
# end

# Example knapsack
n = 10;
a = rand(n);
c = rand(n);
b = 10/2*rand();

# Let's find the JuMP solution
m = knapsack(c,a,b);
m,x = OCT.jump_it(m)
solve(m);
x_vals = getvalue(m[:x])

# TRAIN ONLY OVER X
# Hyperplanes try to guess a and b in the beginning.
# n_samples = 2000;
# fn =  x -> b - sum(a.*x)
# dists = [Distributions.Uniform(0,1) for i=1:n];
# X = reduce(hcat,[rand(dists[i],n_samples) for i=1:n]);
# feas_tree = OCT.learn_constraints(OCT.base_otc(), [fn], X)
# IAI.show_in_browser(feas_tree[1].lnr)
# # We only ever get one split!
#
# # TRAIN OVER X AND B
# offsets = ones(n+1); offsets[end] = offsets[end]*5;
# dists = [Distributions.Uniform(0,offsets[i]) for i=1:n+1];
# fn =  x -> x[end] - sum(a.*x[1:end-1])
# X = reduce(hcat,[rand(dists[i],n_samples) for i=1:n+1]);
# feas_tree = OCT.learn_constraints(OCT.base_otc(), [fn], X)
# IAI.show_in_browser(feas_tree[1].lnr)
# # Still only learns one split!
#
# # SEEMS LIKE WE SHOULD TRAIN OVER A, B, C, X
# n=10; n_samples=2000;
# fn = x -> x[end] - sum(x[1:n].*x[n+1:2*n])
# offsets = ones(3*n+1); offsets[end] = offsets[end]*5;
# dists = [Distributions.Uniform(0,offsets[i]) for i=1:3*n+1];
# X = reduce(hcat,[rand(dists[i],n_samples) for i=1:3*n+1]);
# lnr = OCT.base_otc()
# IAI.set_params!(lnr, max_depth=10)
# IAI.set_params!(lnr, hyperplane_config=(sparsity=1),)
# feas_tree = OCT.learn_constraints(lnr, [fn], X)
# IAI.show_in_browser(feas_tree[1].lnr)


# Finding optimal knapsack solutions...
# n_samples=2000; n = 10;
# as = rand(n_samples, n);
# bs = 0.5 .* rand(Distributions.Gamma(3,2), n_samples);
# cs = rand(n_samples, n);
# xs = zeros(n_samples, n);
# optimum = zeros(n_samples);
# for i=1:n_samples
#     print("Sample", i)
#     m = knapsack_jump(cs[i,:],as[i,:],bs[i]);
#     solve(m);
#     xs[i,:] = getvalue(m[:x]);
#     optimum[i] = sum(cs[i,:].*xs[i,:]);
# end

# coas = cs./as;
# aobs = reduce(hcat, [as[j,:]./bs[j] for j=1:n_samples])';
# sort_indices = reduce(hcat,[sortperm(coas[i,:]) for i=1:n_samples])';
# # # Reordering
# aobs_sorted = reduce(hcat, [aobs[i,sort_indices[i,:]] for i=1:n_samples])';
# coas_sorted = reduce(hcat, [coas[i,sort_indices[i,:]] for i=1:n_samples])';
# cs_sorted = reduce(hcat, [cs[i,sort_indices[i,:]] for i=1:n_samples])';
# xs_sorted = reduce(hcat, [xs[i,sort_indices[i,:]] for i=1:n_samples])';

# Regression for optimal cost
# lnr = OCT.base_otr()
# X = DataFrame(hcat(aobs_sorted, coas_sorted, cs_sorted));
# DataFrames.names!(X,vcat([Symbol('a',i) for i=1:n], [Symbol('b',i) for i=1:n],
# [Symbol('c',i) for i=1:n]));
# Y = DataFrame(y=optimum);
# IAI.fit!(lnr, X, optimum)


# Feature-wise training of OCTS
lnr = OCT.base_otc
lnrs = []
# for i=1:n
#     nl = lnr()
#     IAI.fit!(nl, regressor_data, xs_sorted[:,i])
#     push!(lnrs, nl)
#     IAI.show_in_browser(nl)
# end
