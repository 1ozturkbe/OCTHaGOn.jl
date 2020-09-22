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
          15   15
          16   16
          17   17
          18   18
          19   19
          20   20
          21   21  /;

Parameter t(j)
          a(j);

Variables
           x(i)
           y(j)
           objvar;

t(j) = 10*(aux(j) - 1)/20;
a(j) = 3*exp(-t(j))/20 + 1*exp(-5*t(j))/52 - exp(-2*t(j))*(3*sin(2*t(j)) + 11*cos(2*t(j)))/65;

Equations  e(j)
           e1;

e(j).. abs( x('3')*exp(-t(j)*x('1'))*sin(t(j)*x('2'))/x('2') - a(j) )
     - y(j) =E= 0;
e1.. smax(j,y(j)) - objvar =E= 0;

x.lo('2') = 1e-6;

x.l('1') = 1;
x.l('2') = 1;
x.l('3') = 1;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;