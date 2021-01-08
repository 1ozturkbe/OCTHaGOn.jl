using CSV

"""
    save_solution(gm::GlobalModel; dir::String = SOL_DIR)

Saves the optimal solution of GlobalModel as a CSV.
"""
function save_solution(gm::GlobalModel; name::String = gm.name, dir::String = SOL_DIR)
    open(dir * name * ".txt", "w") do file
        write(file, solution(gm))
    end
end