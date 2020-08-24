struct OCTException <: Exception
    var::String
end

Base.showerror(io::IO, e::OCTException) = print(io, e.var)

