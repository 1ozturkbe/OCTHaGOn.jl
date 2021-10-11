 # Solving the quadrotor trajectory optimization problem. 

 n = 20
 z_des = collect(range(0, n, step=1))
 y_des = z_des .* sin.(z_des*pi/8)
 x_des = z_des .* cos.(z_des*pi/16)

 plot(x_des, y_des, z_des)

 # Defining the quadrotor model
m = Model(CPLEX.Optimizer)
@variable(m, )