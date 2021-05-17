""" Very coarse solution for gm to check for feasibility and an initial starting point. """
function surveysolve(gm::GlobalModel)
    bbcs = [bbl for bbl in gm.bbls if bbl isa BlackBoxClassifier]
    if !isempty(bbcs)
        for bbc in bbcs
            learn_constraint!(bbc)
            update_tree_constraints!(gm, bbc)
        end
    end
    bbrs = [bbl for bbl in gm.bbls if bbl isa BlackBoxRegressor]
    if !isempty(bbrs)
        for bbr in bbrs
            learn_constraint!(bbr, regression_sparsity = 0, max_depth = 6, minbucket = 2*length(bbr.vars))
            update_tree_constraints!(gm, bbr)
        end
    end
    optimize!(gm)
    return
end

"""
    function add_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    function add_relaxation_variables!(gm::GlobalModel, bbls::Array)

Populates relax_var attributes of all substructs. 
"""
function add_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    if bbl.relax_var isa Real
        bbl.relax_var = @variable(gm.model)
        @constraint(gm.model, bbl.relax_var >= 0)  
    end
    if !isempty(bbl.mi_constraints)
        clear_tree_constraints!(gm, bbl)
        @info "Clearing MI constraints from $(bbl.name) due to relaxation."
    end
    if bbl isa BlackBoxLearner
        for ll in bbl.lls
            add_relaxation_variables!(gm, ll)
        end
    end
end

function add_relaxation_variables!(gm::GlobalModel, bbls::Array)
    for bbl in bbls
        add_relaxation_variables!(gm, bbl)
    end
end

add_relaxation_variables!(gm::GlobalModel) = add_relaxation_variables!(gm, gm.bbls)

function clear_relaxation_variables!(gm::GlobalModel, bbl::Union{BlackBoxLearner, LinkedLearner})
    if bbl.relax_var isa JuMP.VariableRef
        is_valid(gm.model, bbl.relax_var) || 
            throw(OCTException("$(bbl.relax_var) is not a valid variable of  $(gm.model)."))
        delete(gm.model, bbl.relax_var)
        bbl.relax_var = 0
    end
    if bbl isa BlackBoxLearner
        for ll in bbl.lls
            clear_relaxation_variables!(gm, ll)
        end
    end
end

function clear_relaxation_variables!(gm::GlobalModel, bbls::Array)
    for bbl in bbls
        clear_relaxation_variables!(gm, bbl)
    end
end

clear_relaxation_variables!(gm::GlobalModel) = clear_relaxation_variables!(gm, gm.bbls)

function relax_objective(gm::GlobalModel, M::Real = 1e5)
    relax_term = 0
    for bbl in gm.bbls
        relax_term += bbl.relax_var
        for ll in bbl.lls
            relax_term += ll.relax_var
        end
    end
    @objective(gm.model, Min, gm.objective + M * relax_term)
    return 
end