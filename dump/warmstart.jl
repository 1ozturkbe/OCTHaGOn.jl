using Random
function fit_warmstart!(lnr, X, y, tree)
  IAI.IAIBase.resetinternalparams!(lnr)
  criterion = IAI.IAIBase.process_criterion(lnr)
  IAI.IAIBase.validate_criterion(criterion, lnr, :criterion)
  data = IAI.IAIBase.Data(tree, X, nothing, y)
  mins, maxs = IAI.IAIBase.get_extrema(data.features.X_numeric,
                                   data.features.n_numeric_orig_features)
  data.features.X_mins .= mins
  data.features.X_maxs .= maxs
  lnr.normalize_X && IAI.IAIBase.normalize_X!(data.features)
  IAI.IAIBase._preprocess!(lnr, data, IAI.IAIBase.gettasktype(lnr))
  lnr.prb_ = IAI.IAIBase.Problem(lnr, data, criterion)

  ls = IAI.OptimalTrees.LocalSearcher(lnr, lnr.prb_)
  IAI.OptimalTrees.init!(ls, lnr)
  Random.seed!(ls.rng_gen, ls.random_seed)

  lnr.all_trees_ = deepcopy(tree.all_trees_)
  for t in 1:length(lnr.all_trees_)
    rng = IAI.IAIBase.RandomStreams.next_stream(ls.rng_gen)
    IAI.OptimalTrees.set_rng!(ls, rng)

    tree = lnr.all_trees_[t]

    if lnr.normalize_X
      IAI.IAIBase.normalize_X!(tree, lnr.prb_.data.features)
    end
    if :normalize_y in fieldnames(typeof(lnr)) && lnr.normalize_y
      IAI.IAIBase.normalize_y!(tree, lnr.prb_.data.target)
    end

    IAI.OptimalTrees.local_search!(tree, ls)

    IAITrees.finalize_capacity!(tree)
    IAI.OptimalTrees.update_tree_fits!(tree, ls.data, ls.evaluators, true)
    lnr.all_trees_[t] = tree

  end
  best_ind = argmin([IAI.OptimalTrees.train_error(lnr, tree)
                     for tree in lnr.all_trees_])
  lnr.tree_ = lnr.all_trees_[best_ind]

  lnr
end
