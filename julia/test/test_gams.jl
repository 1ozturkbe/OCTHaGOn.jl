#=
test_gams:
- Julia version: 1.3.1
- Author: Berk
- Date: 2020-08-21
=#
using GAMSFiles

filename = "../data/ex12.2.1.gms"
parsegams(filename)