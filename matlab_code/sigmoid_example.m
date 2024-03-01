clear all
close all

x = dlarray(-10:0.1:10);

y1 = 1 * sigmoid(x);
y2 = -1 * sigmoid(x);

plot(extractdata(x), extractdata(y1), 'g', extractdata(x), extractdata(y2), 'r');