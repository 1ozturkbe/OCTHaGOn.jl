# File to generate constraints compatible with JuMP
using JuMP
using Gurobi

function add_mio_constraints(lnr, m, x, y, M=1e5)
    """
    Creates a set of MIO constraints from a OptimalTreeLearner
    Arguments:
        lnr:OptimalTreeLearner
        m::JuMP Model
        x::JuMPVariable defined in lnr which are features in lnr
        y::JuMPVariable defined in lnr which is the output of interest
        M::coefficient in bigM formulation
    """
    n_nodes = IAI.get_num_nodes(lnr);
    # Add a binary variable for each leaf
    all_leaves = [i for i=1:n_nodes if IAI.is_leaf(lnr,i)];
    @variable(m, z[1:size(all_leaves,1)], Bin);
    @constraint(m, sum(z) == 1);
    # ADDING CUTS
    for i = 1:size(all_leaves, 1)
        β0 = IAI.get_regression_constant(lnr, all_leaves[i]);
        weights = IAI.get_regression_weights(lnr, all_leaves[i])[1];
        β = [];
        for i=1:size(vks,1)
            if vks[i] in keys(weights)
                append!(β, weights[vks[i]]);
            else
                append!(β, 0.);
            end
        end
        @constraint(m, sum(β.*x) + β0 <= y + M*(1 .-z[i]));
    end
    # ADDING TRUST REGIONS
    for i = 1:size(all_leaves,1)
        # Find all parents
        parents = [all_leaves[i]];
        while IAI.get_depth(lnr, parents[end]) > 0
            append!(parents, IAI.get_parent(lnr, parents[end]))
        end
        parents = parents[2:end];
        for j in parents
            # For each parent, define trust region with binary variables
            threshold = IAI.get_split_threshold(lnr, j);
            if IAI.is_hyperplane_split(lnr, j)
                weights = IAI.get_split_weights(lnr, j);
            else
                feature = IAI.get_split_feature(lnr, j);
                weights = Dict(feature => 1);
            end
            upper = IAI.get_upper_child(lnr, j) in parents
            α = []
            for i=1:size(vks,1)
                if vks[i] in keys(weights)
                    append!(α, weights[vks[i]]);
                else
                    append!(α, 0.);
                end
            end
            if upper
                @constraint(m, threshold <= sum(α.*x) + M*(1-z[i]));
            else
                @constraint(m, threshold + M*(1-z[i]) >= sum(α.*x));
            end
        end
    end
    return m
end