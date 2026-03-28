% Script to calculate average timecost
al = 1.5;

% load(strcat("./result/CI_stockall5", string(100*al)));
load(strcat("./result/CI_stockall5_comp.mat"));

[n_sectors, ~, n_iterations] = size(timecost);

total_time_per_experiment = zeros(n_iterations, 1);

for i = 1:n_iterations
time_matrix = timecost(:, :, i);
total_time_per_experiment(i) = sum(triu(time_matrix, 0), 'all');
end

average_time = mean(total_time_per_experiment);

fprintf('Total time per experiment:\n');
for i = 1:n_iterations
fprintf('Experiment %d: %.4f seconds\n', i, total_time_per_experiment(i));
end
fprintf('\n');
fprintf('Average time over all experiments: %.4f seconds\n', average_time);
fprintf('Standard deviation: %.4f seconds\n', std(total_time_per_experiment));

save('./result/timecost_analysis.mat', 'total_time_per_experiment', 'average_time');

figure;
plot(1:n_iterations, total_time_per_experiment, 'o-', 'LineWidth', 2);
xlabel('Experiment number');
ylabel('Total time (seconds)');
title('Total time per experiment');
grid on;
yline(average_time, '--r', sprintf('Average time: %.4f s', average_time), 'LineWidth', 2);