Sets  i    /1*6/
      j    /1*51/
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
          30   30
          31   21
          32   32
          33   33
          34   34
          35   35
          36   36
          37   37
          38   38
          39   39
          40   40
          41   41
          42   42
          43   43
          44   44
          45   45
          46   46
          47   47
          48   48
          49   49
          50   50
          51   51/;

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
e1.. smax(j,y(j)) - objvar =E= 0;

*x.l('1') = 2;
*x.l('2') = 2;
*x.l('3') = 7;
*x.l('4') = 0;
*x.l('5') = -2;
*x.l('6') = 1;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;