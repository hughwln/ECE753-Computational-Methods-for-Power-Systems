% ====================================================
% author: Yi Hu (yhu28@ncsu.edu)
% date: 2/6/2024
% example code for ECE753 homework, missing data restoration
% ====================================================

clear all
close all

load_temperature = csvread("../data/load_temperature_masked.csv", 1, 1);
% plot yearly load profile, with 5 missing segments
plot(load_temperature(:, 1));

load = reshape(load_temperature(:, 1), 96, 365);
missing_days = [
    31+28+18,                           % 3/18
    31+28+31+30+18,                     % 5/18
    31+28+31+30+31+30+18,               % 7/18
    31+28+31+30+31+30+31+31+18,         % 9/18
    31+28+31+30+31+30+31+31+30+31+18    % 11/18
    ];
nonmissing_days = setdiff(1:365, missing_days);
load_missing_days = load(:, missing_days);
load_nonemissing_days = load(:, nonmissing_days);

temperature = reshape(load_temperature(:, 2), 96, 365);

ests = zeros(5, 16);
% interpolation
for day = 1:5
    x = 1:31:96;
    v = load_missing_days(x, day);
    xq = 1:1:96;
    vq = interp1(x, v, xq, 'linear');
    est = vq(41: 56);        % estimated missing segment
    ests(day, :) = est;
    subplot(2, 3, day);
    plot(x, v, 'o', xq, vq, ':.', xq, load_missing_days(:, day));
end

output = reshape(ests.', 80, 1);
writematrix(output, 'output.csv');