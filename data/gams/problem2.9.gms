Sets  i    /1*4/
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
          11   11  /;

Parameter
          a(j)
         / 1 0.1957
           2 0.1947
           3 0.1735
           4 0.1600
           5 0.0844
           6 0.0627
           7 0.0456
           8 0.0342
           9 0.0323
          10 0.0235
          11 0.0246 /;

Parameter
          u(j)
         / 1 4.0
           2 2.0
           3 1.0
           4 0.5
           5 0.25
           6 0.167
           7 0.125
           8 0.1
           9 0.0833
          10 0.0714
          11 0.0625 /;

Variables
           x(i)
           y(j)
           objvar;

Equations  e(j)
           e1;

e(j).. abs( (x('1')*(power(u(j),2) + x('2')*u(j)))/(power(u(j),2) +
    x('3')*u(j) + x('4')) - a(j) )
     - y(j) =E= 0;
e1.. smax(j,y(j)) - objvar =E= 0;

x.l('1') = 0.25;
x.l('2') = 0.39;
x.l('3') = 0.415;
x.l('4') = 0.39;

Model m / all /;
m.limrow=0; m.limcol=0;

Solve m using DNLP minimizing objvar;