using Random
function fit_warmstart!(lnr, X, y, tree)
  IAIBase.resetinternalparams!(lnr)
  criterion = IAIBase.process_criterion(lnr)
  IAIBase.validate_criterion(criterion, lnr, :criterion)
  data = IAIBase.Data(tree, X, nothing, y)
  mins, maxs = IAIBase.get_extrema(data.features.X_numeric,
                                   data.features.n_numeric_orig_features)
  data.features.X_mins .= mins
  data.features.X_maxs .= maxs
  lnr.normalize_X && IAIBase.normalize_X!(data.features)
  IAIBase._preprocess!(lnr, data, IAIBase.gettasktype(lnr))
  lnr.prb_ = IAIBase.Problem(lnr, data, criterion)

  ls = OptimalTrees.LocalSearcher(lnr, lnr.prb_)
  OptimalTrees.init!(ls, lnr)
  Random.seed!(ls.rng_gen, ls.random_seed)

  lnr.all_trees_ = deepcopy(tree.all_trees_)
  for t in 1:length(lnr.all_trees_)
    rng = IAIBase.RandomStreams.next_stream(ls.rng_gen)
    OptimalTrees.set_rng!(ls, rng)

    tree = lnr.all_trees_[t]

    if lnr.normalize_X
      IAIBase.normalize_X!(tree, lnr.prb_.data.features)
    end
    if :normalize_y in fieldnames(typeof(lnr)) && lnr.normalize_y
      IAIBase.normalize_y!(tree, lnr.prb_.data.target)
    end

    OptimalTrees.local_search!(tree, ls)

    IAITrees.finalize_capacity!(tree)
    OptimalTrees.update_tree_fits!(tree, ls.data, ls.evaluators, true)
    lnr.all_trees_[t] = tree

  end
  best_ind = argmin([OptimalTrees.train_error(lnr, tree)
                     for tree in lnr.all_trees_])
  lnr.tree_ = lnr.all_trees_[best_ind]

  lnr
end
