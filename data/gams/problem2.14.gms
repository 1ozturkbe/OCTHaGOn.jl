Sets  i    /1*5/
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
          21   21/;

Parameter
          t(j);

t(j) = -1 + (aux(j) - 1)/10;

Variables
           x(i)
           y(j)
           objvar;

Equations  e(j)
           e1;

e(j).. abs(  (x('1') + x('2')*t(j)) /
      (1 + x('3')*t(j) + x('4')*power(t(j),2) + x('5')*power(t(j),3)) -
      exp(t(j))
      ) - y(j) =E= 0;
e1.. smax(j,y(j)) - objvar =E= 0;

x.l('1') = 0.5;
x.l('2') = 0;
x.l('3') = 0;
x.l('4') = 0;
x.l('5') = 0;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;