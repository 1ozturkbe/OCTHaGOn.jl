$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         39       38        0        1        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         77       33       44        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*        275       99      176        0
*
*  Solve m using MINLP minimizing objvar;


Variables  objvar,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18
          ,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,b34,b35
          ,b36,b37,b38,b39,b40,b41,b42,b43,b44,b45,b46,b47,b48,b49,b50,b51,b52
          ,b53,b54,b55,b56,b57,b58,b59,b60,b61,b62,b63,b64,b65,b66,b67,b68,b69
          ,b70,b71,b72,b73,b74,b75,b76,b77;

Positive Variables  x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20
          ,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33;

Binary Variables  b34,b35,b36,b37,b38,b39,b40,b41,b42,b43,b44,b45,b46,b47,b48
          ,b49,b50,b51,b52,b53,b54,b55,b56,b57,b58,b59,b60,b61,b62,b63,b64,b65
          ,b66,b67,b68,b69,b70,b71,b72,b73,b74,b75,b76,b77;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36
          ,e37,e38,e39;


e1..    objvar =G= x21;

e2..    x6 =L= 0.0675;

e3..    x8 - x9 + x10 =E= 0;

e4..    x11 - x12 + x13 =E= 0;

e5..    x14 - x15 + x16 =E= 0;

e6..    x17 - x18 + x19 =E= 0;

e7..    x22 - x23 + x24 =E= 0;

e8..    x25 - x26 + x27 =E= 0;

e9..    x28 - x29 + x30 =E= 0;

e10..    x31 - x32 + x33 =E= 0;

e11.. x2**0.29*x9 - x10 =E= 0;

e12.. x3**0.13*x12 - x13 =E= 0;

e13.. x4**0.06*x15 - x16 =E= 0;

e14.. x5**0.15*x18 - x19 =E= 0;

e15.. x2**0.74*x23 - x24 =E= 0;

e16.. x3**0.79*x26 - x27 =E= 0;

e17.. x4**0.71*x29 - x30 =E= 0;

e18.. x5**0.8*x32 - x33 =E= 0;

e19.. b34*x8 + b38*x10 + b42*x11 + b46*x13 + b50*x14 + b54*x16 + b58*x17 + b62*
      x19 - x9 + 0.675*b66 =E= 0;

e20.. b35*x8 + b39*x10 + b43*x11 + b47*x13 + b51*x14 + b55*x16 + b59*x17 + b63*
      x19 - x12 + 0.675*b67 =E= 0;

e21.. b36*x8 + b40*x10 + b44*x11 + b48*x13 + b52*x14 + b56*x16 + b60*x17 + b64*
      x19 - x15 + 0.675*b68 =E= 0;

e22.. b37*x8 + b41*x10 + b45*x11 + b49*x13 + b53*x14 + b57*x16 + b61*x17 + b65*
      x19 - x18 + 0.675*b69 =E= 0;

e23.. b34*x22 + b38*x24 + b42*x25 + b46*x27 + b50*x28 + b54*x30 + b58*x31 + b62
      *x33 - x23 + 0.649*b66 =E= 0;

e24.. b35*x22 + b39*x24 + b43*x25 + b47*x27 + b51*x28 + b55*x30 + b59*x31 + b63
      *x33 - x26 + 0.649*b67 =E= 0;

e25.. b36*x22 + b40*x24 + b44*x25 + b48*x27 + b52*x28 + b56*x30 + b60*x31 + b64
      *x33 - x29 + 0.649*b68 =E= 0;

e26.. b37*x22 + b41*x24 + b45*x25 + b49*x27 + b53*x28 + b57*x30 + b61*x31 + b65
      *x33 - x32 + 0.649*b69 =E= 0;

e27.. b70*x8 + b71*x11 + b72*x14 + b73*x17 - x6 =E= 0;

e28.. b70*x22 + b71*x25 + b72*x28 + b73*x31 - x20 =E= 0;

e29.. b74*x10 + b75*x13 + b76*x16 + b77*x19 - x7 =E= 0;

e30.. b74*x24 + b75*x27 + b76*x30 + b77*x33 - x21 =E= 0;

e31..    b34 + b35 + b36 + b37 + b70 =E= 1;

e32..    b42 + b43 + b44 + b45 + b71 =E= 1;

e33..    b50 + b51 + b52 + b53 + b72 =E= 1;

e34..    b58 + b59 + b60 + b61 + b73 =E= 1;

e35..    b38 + b39 + b40 + b41 + b74 =E= 1;

e36..    b46 + b47 + b48 + b49 + b75 =E= 1;

e37..    b54 + b55 + b56 + b57 + b76 =E= 1;

e38..    b62 + b63 + b64 + b65 + b77 =E= 1;

e39..    b66 + b67 + b68 + b69 =E= 1;

* set non-default bounds
x2.lo = 0.1; x2.up = 0.9;
x3.lo = 0.1; x3.up = 0.9;
x4.lo = 0.1; x4.up = 0.9;
x5.lo = 0.1; x5.up = 0.9;
x6.up = 10;
x7.up = 10;
x8.up = 10;
x9.up = 10;
x10.up = 10;
x11.up = 10;
x12.up = 10;
x13.up = 10;
x14.up = 10;
x15.up = 10;
x16.up = 10;
x17.up = 10;
x18.up = 10;
x19.up = 10;
x20.up = 10;
x21.up = 10;
x22.up = 10;
x23.up = 10;
x24.up = 10;
x25.up = 10;
x26.up = 10;
x27.up = 10;
x28.up = 10;
x29.up = 10;
x30.up = 10;
x31.up = 10;
x32.up = 10;
x33.up = 10;

Model m / all /;

m.limrow=0; m.limcol=0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set MINLP $set MINLP MINLP
Solve m using %MINLP% minimizing objvar;
