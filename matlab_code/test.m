% ====================================================
% author: Yi Hu (yhu28@ncsu.edu)
% date: 2/6/2024
% evaluate the estimation accuracy for ECE753 homework
% ====================================================

clear all
close all

GT = csvread("../data/gt.csv"); % dim:80*1, real data, will be available after due date
M = csvread("output.csv"); % dim:80*1, your result
%M = csvread("../BERT-PIN(Python)/output.csv");

mpe = mean(abs(GT-M)./GT)




% mpe < 3%          10/10
% 3% <= mpe < 5%    9/10
% 5% <= mpe < 10%   8/10
% 10% <= mpe < 20%  7/10
% 20% <= mpe        6/10