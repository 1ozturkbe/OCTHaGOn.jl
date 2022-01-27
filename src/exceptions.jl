struct OCTHaGOnException <: Exception
    var::String
end

Base.showerror(io::IO, e::OCTHaGOnException) = print(io, e.var)

