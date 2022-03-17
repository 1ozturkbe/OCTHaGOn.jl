$offlisting
*  MINLP written by GAMS Convert at 09/01/02 11:06:08
*  
*  Equation counts
*     Total       E       G       L       N       X       C
*        21       1       0      20       0       0       0
*  
*  Variable counts
*                 x       b       i     s1s     s2s      sc      si
*     Total    cont  binary integer    sos1    sos2   scont    sint
*        21       1       0      20       0       0       0       0
*  FX     0       0       0       0       0       0       0       0
*  
*  Nonzero counts
*     Total   const      NL     DLL
*       182     162      20       0
*
*  Solve m using MINLP minimizing objvar;


Variables  i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17,i18,i19
          ,i20,objvar;

Integer Variables i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16,i17
          ,i18,i19,i20;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21;


e1..    8*i1 + 5*i6 + 4*i7 + 6*i12 + 6*i13 + 9*i14 + 5*i19 + i20 =L= 220;

e2..    3*i1 + 4*i2 + 3*i7 + 7*i8 + 4*i13 + 9*i14 + 3*i15 + 2*i20 =L= 175;

e3..    2*i2 + i3 + 6*i8 + 8*i9 + 9*i14 + 9*i15 + 8*i16 =L= 215;

e4..    7*i3 + i4 + 7*i9 + 9*i10 + 2*i15 + 4*i16 + 9*i17 =L= 195;

e5..    4*i4 + 4*i5 + i10 + 3*i11 + 7*i16 + 2*i17 + 8*i18 =L= 145;

e6..    9*i5 + 5*i6 + 5*i11 + 7*i12 + i17 + 4*i18 + 6*i19 =L= 185;

e7..    5*i1 + 5*i6 + 3*i7 + 8*i12 + 5*i13 + 9*i18 + 9*i19 + i20 =L= 225;

e8..    i1 + 9*i2 + 9*i7 + 3*i8 + 9*i13 + 7*i14 + 4*i19 + i20 =L= 215;

e9..    3*i1 + 6*i2 + 3*i3 + 4*i8 + 2*i9 + 6*i14 + 3*i15 + 8*i19 + i20 =L= 175;

e10..    i2 + 2*i3 + 8*i4 + 4*i9 + i10 + 9*i15 + 6*i16 =L= 155;

e11..    9*i3 + 3*i4 + 6*i5 + i10 + 6*i11 + 9*i16 + 8*i17 =L= 210;

e12..    6*i4 + 3*i5 + 3*i6 + 6*i11 + 3*i12 + 8*i17 + 9*i18 =L= 190;

e13..    9*i5 + 8*i6 + 2*i7 + 7*i12 + 8*i13 + 4*i18 + 3*i19 =L= 205;

e14..    4*i1 + 6*i6 + 9*i7 + i8 + 6*i13 + 9*i14 + 8*i19 + 6*i20 =L= 245;

e15..    7*i1 + 3*i2 + 7*i7 + 4*i8 + 2*i9 + i14 + 3*i15 + 5*i20 =L= 160;

e16..    7*i2 + 9*i3 + 7*i8 + 9*i9 + 5*i10 + 2*i15 + 6*i16 =L= 225;

e17..    6*i3 + 9*i4 + 8*i9 + 4*i10 + 2*i11 + 6*i16 + 4*i17 =L= 195;

e18..    5*i4 + 5*i5 + 7*i10 + 8*i11 + 9*i12 + 8*i17 + 6*i18 =L= 240;

e19..    7*i5 + 5*i6 + 6*i11 + 2*i12 + 8*i13 + 6*i18 + 9*i19 =L= 215;

e20..    i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + i10 + i11 + i12 + i13
       + i14 + i15 + i16 + i17 + i18 + i19 + i20 =L= 200;

e21..  - (0.00055*i1*i1 - 0.0583*i1 + 0.0019*i2*i2 - 0.2318*i2 + 0.0002*i3*i3
       + 0.0108*i3 + 0.00095*i4*i4 - 0.1634*i4 + 0.0046*i5*i5 - 0.138*i5 + 
      0.0035*i6*i6 + 0.357*i6 + 0.00315*i7*i7 - 0.1953*i7 + 0.00475*i8*i8 + 
      0.361*i8 + 0.0048*i9*i9 - 0.1824*i9 + 0.003*i10*i10 + 0.162*i10 + 0.00265
      *i11*i11 - 0.4346*i11 + 0.0017*i12*i12 - 0.1054*i12 + 0.0012*i13*i13 + 
      0.2376*i13 + 0.00295*i14*i14 - 0.0059*i14 + 0.00315*i15*i15 + 0.189*i15
       + 0.0021*i16*i16 - 0.0252*i16 + 0.00225*i17*i17 - 0.099*i17 + 0.0034*i18
      *i18 - 0.3604*i18 + 0.001*i19*i19 + 0.022*i19 + 0.00305*i20*i20 - 0.3294*
      i20) + objvar =E= 0;

* set non default bounds

i1.up = 100; 
i2.up = 100; 
i3.up = 100; 
i4.up = 100; 
i5.up = 100; 
i6.up = 100; 
i7.up = 100; 
i8.up = 100; 
i9.up = 100; 
i10.up = 100; 
i11.up = 100; 
i12.up = 100; 
i13.up = 100; 
i14.up = 100; 
i15.up = 100; 
i16.up = 100; 
i17.up = 100; 
i18.up = 100; 
i19.up = 100; 
i20.up = 100; 

$if set nostart $goto modeldef
* set non default levels


* set non default marginals


$label modeldef
Model m / all /;

m.limrow=0; m.limcol=0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set MINLP $set MINLP MINLP
Solve m using %MINLP% minimizing objvar;
