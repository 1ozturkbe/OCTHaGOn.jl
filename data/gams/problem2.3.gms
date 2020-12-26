Sets  i    /1*2/
      j    /1*2/
;

Variables
           x(i)
           y(j)
           objvar;

Equations  e1
           e2
           e3;

e1.. power( x('1')
     - sqrt(power(x('1'),2) + power(x('2'),2))*
     cos( sqrt(power(x('1'),2) + power(x('2'),2)) ),2 ) +
     0.0005*( power(x('1'),2) + power(x('2'),2) ) - y('1') =E= 0;
e2.. power( x('2')
     - sqrt(power(x('1'),2) + power(x('2'),2))*
     sin( sqrt(power(x('1'),2) + power(x('2'),2)) ),2 ) +
     0.0005*( power(x('1'),2) + power(x('2'),2) ) - y('2') =E= 0;
e3.. smax(j,y(j)) - objvar =E= 0;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;