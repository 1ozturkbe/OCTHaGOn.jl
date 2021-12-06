$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         66       11       10       45        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         63       23       40        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*        243      238        5        0
*
*  Solve m using MINLP minimizing objvar;


Variables  x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19
          ,x20,x21,x22,b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33,b34,b35,b36
          ,b37,b38,b39,b40,b41,b42,b43,b44,b45,b46,b47,b48,b49,b50,b51,b52,b53
          ,b54,b55,b56,b57,b58,b59,b60,b61,b62,objvar;

Positive Variables  x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x21,x22;

Binary Variables  b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33,b34,b35,b36,b37
          ,b38,b39,b40,b41,b42,b43,b44,b45,b46,b47,b48,b49,b50,b51,b52,b53,b54
          ,b55,b56,b57,b58,b59,b60,b61,b62;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36
          ,e37,e38,e39,e40,e41,e42,e43,e44,e45,e46,e47,e48,e49,e50,e51,e52,e53
          ,e54,e55,e56,e57,e58,e59,e60,e61,e62,e63,e64,e65,e66;


e1..  2*x21 + 2*x22 =L= objvar;

e2..  - x1 - x11 + x21 =G= 0;

e3..  - x2 - x12 + x21 =G= 0;

e4..  - x3 - x13 + x21 =G= 0;

e5..  - x4 - x14 + x21 =G= 0;

e6..  - x5 - x15 + x21 =G= 0;

e7..  - x6 - x16 + x22 =G= 0;

e8..  - x7 - x17 + x22 =G= 0;

e9..  - x8 - x18 + x22 =G= 0;

e10..  - x9 - x19 + x22 =G= 0;

e11..  - x10 - x20 + x22 =G= 0;

e12.. 40/x16 - x11 =L= 0;

e13.. 50/x17 - x12 =L= 0;

e14.. 60/x18 - x13 =L= 0;

e15.. 35/x19 - x14 =L= 0;

e16.. 75/x20 - x15 =L= 0;

e17..    x1 - x2 + x11 + 69*b23 =L= 69;

e18..    x1 - x3 + x11 + 69*b24 =L= 69;

e19..    x1 - x4 + x11 + 69*b25 =L= 69;

e20..    x1 - x5 + x11 + 69*b26 =L= 69;

e21..    x2 - x3 + x12 + 79*b27 =L= 79;

e22..    x2 - x4 + x12 + 79*b28 =L= 79;

e23..    x2 - x5 + x12 + 79*b29 =L= 79;

e24..    x3 - x4 + x13 + 89*b30 =L= 89;

e25..    x3 - x5 + x13 + 89*b31 =L= 89;

e26..    x4 - x5 + x14 + 64*b32 =L= 64;

e27..  - x1 + x2 + x12 + 79*b33 =L= 79;

e28..  - x1 + x3 + x13 + 89*b34 =L= 89;

e29..  - x1 + x4 + x14 + 64*b35 =L= 64;

e30..  - x1 + x5 + x15 + 104*b36 =L= 104;

e31..  - x2 + x3 + x13 + 89*b37 =L= 89;

e32..  - x2 + x4 + x14 + 64*b38 =L= 64;

e33..  - x2 + x5 + x15 + 104*b39 =L= 104;

e34..  - x3 + x4 + x14 + 64*b40 =L= 64;

e35..  - x3 + x5 + x15 + 104*b41 =L= 104;

e36..  - x4 + x5 + x15 + 104*b42 =L= 104;

e37..    x6 - x7 + x16 + 69*b43 =L= 69;

e38..    x6 - x8 + x16 + 69*b44 =L= 69;

e39..    x6 - x9 + x16 + 69*b45 =L= 69;

e40..    x6 - x10 + x16 + 69*b46 =L= 69;

e41..    x7 - x8 + x17 + 79*b47 =L= 79;

e42..    x7 - x9 + x17 + 79*b48 =L= 79;

e43..    x7 - x10 + x17 + 79*b49 =L= 79;

e44..    x8 - x9 + x18 + 89*b50 =L= 89;

e45..    x8 - x10 + x18 + 89*b51 =L= 89;

e46..    x9 - x10 + x19 + 64*b52 =L= 64;

e47..  - x6 + x7 + x17 + 79*b53 =L= 79;

e48..  - x6 + x8 + x18 + 89*b54 =L= 89;

e49..  - x6 + x9 + x19 + 64*b55 =L= 64;

e50..  - x6 + x10 + x20 + 104*b56 =L= 104;

e51..  - x7 + x8 + x18 + 89*b57 =L= 89;

e52..  - x7 + x9 + x19 + 64*b58 =L= 64;

e53..  - x7 + x10 + x20 + 104*b59 =L= 104;

e54..  - x8 + x9 + x19 + 64*b60 =L= 64;

e55..  - x8 + x10 + x20 + 104*b61 =L= 104;

e56..  - x9 + x10 + x20 + 104*b62 =L= 104;

e57..    b23 + b33 + b43 + b53 =E= 1;

e58..    b24 + b34 + b44 + b54 =E= 1;

e59..    b25 + b35 + b45 + b55 =E= 1;

e60..    b26 + b36 + b46 + b56 =E= 1;

e61..    b27 + b37 + b47 + b57 =E= 1;

e62..    b28 + b38 + b48 + b58 =E= 1;

e63..    b29 + b39 + b49 + b59 =E= 1;

e64..    b30 + b40 + b50 + b60 =E= 1;

e65..    b31 + b41 + b51 + b61 =E= 1;

e66..    b32 + b42 + b52 + b62 =E= 1;

* set non-default bounds
x1.up = 29;
x2.up = 29;
x3.up = 29;
x4.up = 29;
x5.up = 29;
x6.up = 29;
x7.up = 29;
x8.up = 29;
x9.up = 29;
x10.up = 29;
x11.lo = 1; x11.up = 40;
x12.lo = 1; x12.up = 50;
x13.lo = 1; x13.up = 60;
x14.lo = 1; x14.up = 35;
x15.lo = 1; x15.up = 75;
x16.lo = 1; x16.up = 40;
x17.lo = 1; x17.up = 50;
x18.lo = 1; x18.up = 60;
x19.lo = 1; x19.up = 35;
x20.lo = 1; x20.up = 75;
x21.up = 30;
x22.up = 30;

Model m / all /;

m.limrow=0; m.limcol=0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set MINLP $set MINLP MINLP
Solve m using %MINLP% minimizing objvar;
