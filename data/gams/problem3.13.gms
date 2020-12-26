Sets  i    /1*5/
      j    /1*10/
;

Alias (i,ii)

table   A(j,i)
            1       2       3       4       5
   1      -16       2       0       1       0
   2        0      -2       0       4       2
   3     -3.5       0       2       0       0
   4        0      -2       0      -4      -1
   5        0      -9      -2       1    -2.8
   6        2       0      -4       0       0
   7       -1      -1      -1      -1      -1
   8       -1      -2      -3      -2      -1
   9        1       2       3       4       5
  10        1       1       1       1       1
;

table   C(i,ii)
            1       2       3       4       5
   1       30     -20     -10      32     -10
   2      -20      39      -6     -31      32
   3      -10      -6      10      -6     -10
   4       32     -31      -6      39     -20
   5      -10      32     -10     -20      30
;

Parameter b(j)
         / 1   -40
           2   -2
           3   -0.25
           4   -4
           5   -4
           6   -1
           7   -40
           8   -60
           9   5
          10   1/;

Parameter d(i)
         / 1   4
           2   8
           3  10
           4   6
           5   2/;

Parameter e(i)
         / 1  -15
           2  -27
           3  -36
           4  -18
           5  -12/;

Variables
           x(i)
           y
           f
           objvar;

Equations  e1
           e2
           fone
           ftwo;

e2(j)..   y =g=  b(j) - sum(i, A(j,i)*x(i)) ;

fone..      f =g= 0;
ftwo..      f =g= y;

e1..   sum(i,d(i)*power(x(i),3)) +
       sum(i,sum(ii,C(i,ii)*x(i)*x(ii))) +
       sum(i,e(i)*x(i)) + 50*f
       - objvar =E= 0;

x.l('1') = 0;
x.l('2') = 0;
x.l('3') = 0;
x.l('4') = 0;
x.l('5') = 0;

x.lo('1') = 0;
x.lo('2') = 0;
x.lo('3') = 0;
x.lo('4') = 0;
x.lo('5') = 0;

x.up('1') = 100;
x.up('2') = 100;
x.up('3') = 100;
x.up('4') = 100;
x.up('5') = 100;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;