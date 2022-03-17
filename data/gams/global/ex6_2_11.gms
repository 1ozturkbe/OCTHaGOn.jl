*  NLP written by GAMS Convert at 07/19/01 13:39:43
*  
*  Equation counts
*     Total       E       G       L       N       X
*         2       2       0       0       0       0
*  
*  Variable counts
*                 x       b       i     s1s     s2s      sc      si
*     Total    cont  binary integer    sos1    sos2   scont    sint
*         4       4       0       0       0       0       0       0
*  FX     0       0       0       0       0       0       0       0
*  
*  Nonzero counts
*     Total   const      NL     DLL
*         7       4       3       0
*
*  Solve m using NLP minimizing objvar;


Variables  objvar,x2,x3,x4;

Equations  e1,e2;


e1..  - ((15.3261663216011*x2 + 23.2043471859416*x3 + 6.69678129464404*x4)*log(
     2.1055*x2 + 3.1878*x3 + 0.92*x4) + 1.04055250396734*x2 - 2.24199441248417*
     x3 + 3.1618173099828*x4 + 6.4661663216011*x2*log(x2/(2.1055*x2 + 3.1878*x3
      + 0.92*x4)) + 12.2043471859416*x3*log(x3/(2.1055*x2 + 3.1878*x3 + 0.92*x4
     )) + 0.696781294644034*x4*log(x4/(2.1055*x2 + 3.1878*x3 + 0.92*x4)) + 9.86
     *x2*log(x2/(1.972*x2 + 2.4*x3 + 1.4*x4)) + 12*x3*log(x3/(1.972*x2 + 2.4*x3
      + 1.4*x4)) + 7*x4*log(x4/(1.972*x2 + 2.4*x3 + 1.4*x4)) + (1.972*x2 + 2.4*
     x3 + 1.4*x4)*log(1.972*x2 + 2.4*x3 + 1.4*x4) + 1.972*x2*log(x2/(1.972*x2
      + 0.283910843616504*x3 + 3.02002220174195*x4)) + 2.4*x3*log(x3/(
     1.45991339466884*x2 + 2.4*x3 + 0.415073537580851*x4)) + 1.4*x4*log(x4/(
     0.602183324335333*x2 + 0.115623371371275*x3 + 1.4*x4)) - 17.2981663216011*
     x2*log(x2) - 25.6043471859416*x3*log(x3) - 8.09678129464404*x4*log(x4))
      + objvar =E= 0;

e2..    x2 + x3 + x4 =E= 1;

* set non default bounds

x2.lo = 1E-6; x2.up = 1; 
x3.lo = 1E-6; x3.up = 1; 
x4.lo = 1E-6; x4.up = 1; 

* set non default levels

x2.l = 0.00565; 
x3.l = 0.99054; 
x4.l = 0.00381; 

* set non default marginals


Model m / all /;

m.limrow=0; m.limcol=0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

Solve m using NLP minimizing objvar;
