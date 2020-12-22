Sets  i    /1*3/
      j    /1*21/
;

Parameter aux(j)
         / 1   1
           2   2
           3   3
           4   4
           5   5
           6   6
           7   7
           8   8
           9   9
          10   10
          11   11
          12   12
          13   13
          14   14
          15   15  /;

Parameter u(j)
          v(j)
          w(j);

Parameter
          a(j)
         / 1 0.14
           2 0.18
           3 0.22
           4 0.25
           5 0.29
           6 0.32
           7 0.35
           8 0.39
           9 0.37
          10 0.58
          11 0.73
          12 0.96
          13 1.34
          14 2.10
          15 4.39 /;

Variables
           x(i)
           y(j)
           z
           objvar;

u(j) = aux(j);
v(j) = 16 - aux(j);
w(j) = min(u(j),v(j));

Equations  e(j)
           e1
           z1(j)
           z2(j);

e(j).. x('1') + u(j) / (v(j)*x('2') + w(j)*x('3')) - a(j)
     - y(j) =E= 0;
z1(j).. z =g= y(j);
z2(j).. z =g= -y(j);
e1.. z - objvar =E= 0;

x.l('1') = 1;
x.l('2') = 1;
x.l('3') = 1;
x.lo('1') = -100;
x.lo('2') = -100;
x.lo('3') = -100;
x.up('1') = 100;
x.up('2') = 100;
x.up('3') = 100;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;