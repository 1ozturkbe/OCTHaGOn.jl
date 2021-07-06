include("../load.jl")
include("../algorithms.jl")

# Loading BARON examples (that haven't been loaded yet)
for i in ["nlp1.jl", "nlp2.jl", "nlp3.jl"]
    include(OCT.BARON_DIR * i)
end

BARON_PROBLEMS = [minlp, nlp1, nlp2, nlp3]

bms = [p(false) for p in BARON_PROBLEMS]
optimize_and_time!.(bms)

gms = [p(true) for p in BARON_PROBLEMS]
optimize_and_time!.(gms)

# save_fit.(gms)
# load_fit.(gms)