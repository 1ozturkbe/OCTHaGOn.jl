// process synthes1
// mixed-integer nonlinear program

// Source: Test problem 1 (Synthesis of processing system) in 
// M. Duran & I.E. Grossmann,
// "An outer approximation algorithm for a class of mixed integer nonlinear
//  programs", Mathematical Programming 36, pp. 307-339, 1986.

// Number of variables:   6 (3 binary variables)  
// Number of constraints: 6
// Objective nonlinear
// Nonlinear constraints

BINARY_VARIABLES y1, y2, y3;

POSITIVE_VARIABLES x1, x2, x3;

UPPER_BOUNDS{
x1: 2 ;
x2: 2 ;
x3: 1 ;
}

EQUATIONS c1, c2, c3, c4, c5, c6;

c1: 0.8*log(x2 + 1) + 0.96*log(x1 - x2 + 1) - 0.8*x3 >= 0 ;
c2: log(x2 + 1) + 1.2*log(x1 - x2 + 1) - x3 - 2*y3 >= -2 ;
c3: x2 - x1 <= 0 ;
c4: x2 - 2*y1 <= 0 ;
c5: x1 - x2 - 2*y2 <= 0 ;
c6: y1 + y2 <= 1 ;

OBJ: minimize 
     5*y1 + 6*y2 + 8*y3 + 10*x1 - 7*x3 - 18*log(x2 + 1) 
     - 19.2*log(x1 - x2 + 1) + 10;
