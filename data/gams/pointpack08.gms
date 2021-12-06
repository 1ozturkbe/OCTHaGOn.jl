$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         36        0        0       36        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         17       17        0        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*        156       44      112        0
*
*  Solve m using NLP maximizing obj;


Variables  x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,obj,objvar;

Positive Variables  x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36,e37;


e1.. 2*x1*x2 - x1*x1 - x2*x2 - x9*x9 + 2*x9*x10 - x10*x10 + obj =L= 0;

e2.. 2*x1*x3 - x1*x1 - x3*x3 - x9*x9 + 2*x9*x11 - x11*x11 + obj =L= 0;

e3.. 2*x1*x4 - x1*x1 - x4*x4 - x9*x9 + 2*x9*x12 - x12*x12 + obj =L= 0;

e4.. 2*x1*x5 - x1*x1 - x5*x5 - x9*x9 + 2*x9*x13 - x13*x13 + obj =L= 0;

e5.. 2*x1*x6 - x1*x1 - x6*x6 - x9*x9 + 2*x9*x14 - x14*x14 + obj =L= 0;

e6.. 2*x1*x7 - x1*x1 - x7*x7 - x9*x9 + 2*x9*x15 - x15*x15 + obj =L= 0;

e7.. 2*x1*x8 - x1*x1 - x8*x8 - x9*x9 + 2*x9*x16 - x16*x16 + obj =L= 0;

e8.. 2*x2*x3 - x2*x2 - x3*x3 - x10*x10 + 2*x10*x11 - x11*x11 + obj =L= 0;

e9.. 2*x2*x4 - x2*x2 - x4*x4 - x10*x10 + 2*x10*x12 - x12*x12 + obj =L= 0;

e10.. 2*x2*x5 - x2*x2 - x5*x5 - x10*x10 + 2*x10*x13 - x13*x13 + obj =L= 0;

e11.. 2*x2*x6 - x2*x2 - x6*x6 - x10*x10 + 2*x10*x14 - x14*x14 + obj =L= 0;

e12.. 2*x2*x7 - x2*x2 - x7*x7 - x10*x10 + 2*x10*x15 - x15*x15 + obj =L= 0;

e13.. 2*x2*x8 - x2*x2 - x8*x8 - x10*x10 + 2*x10*x16 - x16*x16 + obj =L= 0;

e14.. 2*x3*x4 - x3*x3 - x4*x4 - x11*x11 + 2*x11*x12 - x12*x12 + obj =L= 0;

e15.. 2*x3*x5 - x3*x3 - x5*x5 - x11*x11 + 2*x11*x13 - x13*x13 + obj =L= 0;

e16.. 2*x3*x6 - x3*x3 - x6*x6 - x11*x11 + 2*x11*x14 - x14*x14 + obj =L= 0;

e17.. 2*x3*x7 - x3*x3 - x7*x7 - x11*x11 + 2*x11*x15 - x15*x15 + obj =L= 0;

e18.. 2*x3*x8 - x3*x3 - x8*x8 - x11*x11 + 2*x11*x16 - x16*x16 + obj =L= 0;

e19.. 2*x4*x5 - x4*x4 - x5*x5 - x12*x12 + 2*x12*x13 - x13*x13 + obj =L= 0;

e20.. 2*x4*x6 - x4*x4 - x6*x6 - x12*x12 + 2*x12*x14 - x14*x14 + obj =L= 0;

e21.. 2*x4*x7 - x4*x4 - x7*x7 - x12*x12 + 2*x12*x15 - x15*x15 + obj =L= 0;

e22.. 2*x4*x8 - x4*x4 - x8*x8 - x12*x12 + 2*x12*x16 - x16*x16 + obj =L= 0;

e23.. 2*x5*x6 - x5*x5 - x6*x6 - x13*x13 + 2*x13*x14 - x14*x14 + obj =L= 0;

e24.. 2*x5*x7 - x5*x5 - x7*x7 - x13*x13 + 2*x13*x15 - x15*x15 + obj =L= 0;

e25.. 2*x5*x8 - x5*x5 - x8*x8 - x13*x13 + 2*x13*x16 - x16*x16 + obj =L= 0;

e26.. 2*x6*x7 - x6*x6 - x7*x7 - x14*x14 + 2*x14*x15 - x15*x15 + obj =L= 0;

e27.. 2*x6*x8 - x6*x6 - x8*x8 - x14*x14 + 2*x14*x16 - x16*x16 + obj =L= 0;

e28.. 2*x7*x8 - x7*x7 - x8*x8 - x15*x15 + 2*x15*x16 - x16*x16 + obj =L= 0;

e29..  - x9 + x10 =L= 0;

e30..  - x1 + x2 =L= 0;

e31..  - x2 + x3 =L= 0;

e32..  - x3 + x4 =L= 0;

e33..  - x4 + x5 =L= 0;

e34..  - x5 + x6 =L= 0;

e35..  - x6 + x7 =L= 0;

e36..  - x7 + x8 =L= 0;

e37..  objvar =G= -obj;

* set non-default bounds
x1.lo = 0.5; x1.up = 1;
x2.lo = 0.5; x2.up = 1;
x3.lo = 0.5; x3.up = 1;
x4.lo = 0.5; x4.up = 1;
x5.up = 1;
x6.up = 1;
x7.up = 1;
x8.up = 1;
x9.up = 1;
x10.up = 1;
x11.up = 1;
x12.up = 1;
x13.up = 1;
x14.up = 1;
x15.up = 1;
x16.up = 1;

Model m / all /;

m.limrow=0; m.limcol=0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set NLP $set NLP NLP
Solve m using %NLP% minimizing objvar;
