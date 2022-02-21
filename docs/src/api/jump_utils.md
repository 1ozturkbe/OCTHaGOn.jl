# JuMP and MOI utilities

These functions act on JuMP and MathOptInterface structures to be able to implement OCTHaGOn. 

```@docs
bound!
bounded_aux
classify_constraints
distance_to_set
fetch_variable
get_bounds
get_constant
get_unbounds
linearize_objective!
restrict_to_set
vars_from_constraint
```

The following functions extend JuMP functions to act on OCTHaGOn's structs. 

```@docs
all_variables
set_optimizer
```