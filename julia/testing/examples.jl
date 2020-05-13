function example1()
    obj(x) = 0.5*exp(x[1]-x[2]) - exp(x[1]) - 5*exp(-x[2]);
    g1(x) = 100 - exp(x[2]-x[3]) - exp(x[2]) - 0.05*exp(x[1]+x[3]);
    constraints = [g1]
    lbs = log.([70, 1, 0.5]);
    ubs = log.([150, 20, 21]);
    return obj, constraints, lbs, ubs
end
