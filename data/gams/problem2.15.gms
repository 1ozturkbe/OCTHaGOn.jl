Sets  i    /1*5/
      j    /1*30/
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
          21   21
          22   22
          23   23
          24   24
          25   25
          26   26
          27   27
          28   28
          29   29
          30   30/;

Parameter
          t(j)
          a(j);

t(j) = -1 + 2*(aux(j) - 1)/29;
a(j) = sqrt( power(8*t(j) - 1,2) + 1 ) * arctan(8*t(j)) / (8*t(j));

Variables
           x(i)
           y(j)
           objvar;

Equations  e(j)
           e1;

e(j).. abs(  (x('1') + x('2')*t(j) + x('3')*power(t(j),2)) /
      ( 1 + x('4')*t(j) + x('5')*power(t(j),2) ) -
      a(j)
      ) - y(j) =E= 0;
e1.. smax(j,y(j)) =L= objvar;

x.l('1') = 0;
x.l('2') = -1;
x.l('3') = 10;
x.l('4') = 1;
x.l('5') = 10;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;