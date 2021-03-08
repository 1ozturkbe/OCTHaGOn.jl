Sets  i    /1*9/
      j    /1*41/
;

$offdigit

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
          41   41/;

Parameter
          t(j)
         / 1        0
           2        0.01
           3        0.02
           4        0.03
           5        0.04
           6        0.05
           7        0.07
           8        0.1
           9        0.13
          10        0.16
          11        0.19
          12        0.22
          13        0.25
          14        0.28
          15        0.31
          16        0.34
          17        0.37
          18        0.4
          19        0.43
          20        0.46
          21        0.5
          22        0.54
          23        0.57
          24        0.6
          25        0.63
          26        0.66
          27        0.69
          28        0.72
          29        0.75
          30        0.78
          31        0.81
          32        0.84
          33        0.87
          34        0.9
          35        0.93
          36        0.95
          37        0.96
          38        0.97
          39        0.98
          40        0.99
          41        1 /;


Parameter
          auxy(j)
          auxcos(j)
          auxsin(j);

Scalar
          pi /3.14159265358979324/;

auxy(j) = abs(1 - 2*t(j));
auxcos(j) = cos(pi*t(j));
auxsin(j) = sin(pi*t(j));

Variables
           x(i)
           y(j)
           objvar;

Equations  e(j)
           e1;

e(j).. abs( x('9')*sqrt(
       ( power(x('1') + (1 + x('2'))*auxcos(j),2) +
         power( (1 - x('2'))*auxsin(j),2) ) /
       ( power(x('3') + (1 + x('4'))*auxcos(j),2) +
         power( (1 - x('4'))*auxsin(j),2) )  ) * sqrt(
       ( power(x('5') + (1 + x('6'))*auxcos(j),2) +
         power( (1 - x('6'))*auxsin(j),2) ) /
       ( power(x('7') + (1 + x('8'))*auxcos(j),2) +
         power( (1 - x('8'))*auxsin(j),2) )  )
        - auxy(j)
      ) - y(j) =E= 0;
e1.. smax(j,y(j)) =L= objvar;

x.l('1') = 0;
x.l('2') = 1;
x.l('3') = 0;
x.l('4') = -0.15;
x.l('5') = 0;
x.l('6') = -0.68;
x.l('7') = 0;
x.l('8') = -0.72;
x.l('9') = 0.37;

Model m / all /;
m.limrow=0; m.limcol=0;

m.optca=1e-3; m.optcr=1e-3;

Solve m using DNLP minimizing objvar;