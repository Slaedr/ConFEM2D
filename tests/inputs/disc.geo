
g = DefineNumber[ 0.075, Name "Parameters/g" ];
r = 1.0;

Point(1) = {0, 0, 0, g};
//+
Point(2) = {r, 0, 0, g};
//+
Point(3) = {0, r, 0, g};
//+
Point(4) = {-r, 0, 0, g};
//+
Point(5) = {0, -r, 0, g};
//+
Circle(1) = {2, 1, 3};
//+
Circle(2) = {3, 1, 4};
//+
Circle(3) = {4, 1, 5};
//+
Circle(4) = {5, 1, 2};

Line Loop(6) = {1,2,3,4};
//+
Plane Surface(7) = {6};
//+
Physical Line(2) = {1, 2, 3, 4};
//+
Physical Surface(8) = {7};
