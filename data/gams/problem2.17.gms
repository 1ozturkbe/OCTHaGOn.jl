Sets  i    /1*6/
      j    /1*11/
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
          11   11/;

Parameter t(j)
         / 1   0.5
           2   0.6
           3   0.7
           4   0.77
           5   0.9
           6   1.0
           7   1.1
           8   1.23
           9   1.3
          10   1.4
          11   1.5/;

Parameter
          t(j)
          a(j);

t(j) = 0.1*(aux(j) - 1);
a(j) = 0.5*exp(-t(j)) - exp(-2*t(j)) + 0.5*exp(-3*t(j)) +
      1.5*exp(-1.5*t(j))*sin(7*t(j)) + exp(-2.5*t(j))*sin(5*t(j));

Variables
           x(i)
           y(j)
           objvar;

Equations  e(j)
           e1;

e(j).. abs(  x('1')*exp(-x('2')*t(j))*cos(x('3')*t(j) + x('4')) +
       x('5')*exp(-x('6')*t(j)) - a(j)
      ) - y(j) =E= 0;
e1.. smax(j,y(j)) =L= objvar;

*x.l('1') = 2;
*x.l('2') = 2;
*x.l('3') = 7;
*x.l('4') = 0;
*x.l('5') = -2;
*x.l('6') = 1;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;