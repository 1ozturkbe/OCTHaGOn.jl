using CSV

"""
    save_solution(gm::GlobalModel; dir::String = SAVE_DIR)

Saves the optimal solution of GlobalModel as a CSV.
"""
function save_solution(gm::GlobalModel; name::String = gm.name, dir::String = SAVE_DIR)
    open(dir * name * ".txt", "w") do file
        write(file, solution(gm))
    end
end