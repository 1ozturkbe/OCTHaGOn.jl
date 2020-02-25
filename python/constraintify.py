import pandas as pd
import numpy as np
import interpretableai as iai

vks = ["Re", "thick", "M", "C_L"];
X = pd.read_csv("../data/airfoil/airfoil_X.csv", header=None)
y = pd.read_csv("../data/airfoil/airfoil_Y.csv", header=None)
X = np.array(X);
y = np.array(y);

Re = np.linspace(10000,35000, num=6, endpoint=True);
thick = np.array([0.100,0.110,0.120,0.130,0.140,0.145]);
M = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);
cl = np.linspace(0.35, 0.70, num=8, endpoint=True);
(train_X, train_y), (test_X, test_y) = iai.split_data('regression', X, y, seed=1)

grid = iai.GridSearch(iai.OptimalTreeRegressor(random_seed=1),
    max_depth=[2,3,4], cp=[0.01, 0.05, 0.001], 
)
grid.fit(train_X, train_y, test_X, test_y)
lnr = grid.get_learner()

from julia import Julia as j

j.include("../julia/gen_constraints.jl")