h = 0.05;
Point(1) = {0, 0, 0, h};
Point(2) = {1, 0, 0, h};
Point(3) = {1, 1, 0, h};
Point(4) = {0, 1, 0, h};
Point(5) = {0.25, 0.25, 0, h};
Point(6) = {0.75, 0.25, 0, h};
Point(7) = {0.75, 0.75, 0, h};
Point(8) = {0.25, 0.75, 0, h};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line Loop(6) = {1, 2, 3, 4};
Line Loop(7) = {5, 6, 7, 8};
Plane Surface(6) = {6, 7};
Physical Line(2) = {1, 2, 3, 4};
Physical Line(4) = {5, 6, 7, 8};
Physical Surface(7) = {6};
