#=
nonlinearize:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-11-24
=#

vars = flat(expr_vars)
aux_vars = [Symbol("aux$(i)") for i=1:length(vars)]
varmap = get_varmap(expr_vars, vars)
lhs = :(($(aux_vars)...))
vars_copy = deepcopy(expr.args[1].args)
expr_copy = deepcopy(expr.args[2])

for i = 1:length(expr_vars)
    tuple_idxs = findall(x -> x[1] == i, varmap)
    if length(tuple_idxs) == 1 # Scalar inputs to the function
        expr_copy = substitute(expr_copy, vars_copy[i]=> aux_vars[tuple_idxs[1]])
    else
        expr_copy = substitute(expr_copy, vars_copy[i] => aux_vars[tuple_idxs])
    end
end
fn = :($(lhs) -> $(expr_copy))