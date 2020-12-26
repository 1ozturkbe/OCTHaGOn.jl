Sets  i    /1*2/
      j    /1*3/
;

Variables
           x(i)
           y(j)
           objvar;

Equations  e1
           e2
           e3
           e4;

e1.. 5*sqrt( 9*power(x('1'),2) + 16*power(x('2'),2) )
     - y('1') =E= 0;
e2.. 9*x('1') + 16*abs(x('2'))
     - y('2') =E= 0;
e3.. 9*x('1') + 16*abs(x('2')) - power(x('1'),9)
     - y('3') =E= 0;
e4.. y('1')*(x('1') >= abs(x('2'))) +
     y('2')*( (0 < x('1')) and ( x('1') < abs(x('2')) ) ) +
     y('3')*(x('1') <= 0) - objvar =E= 0;

x.l('1') = 3;
x.l('2') = 2;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;