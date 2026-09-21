%   'euclidean'  - sqrt(sum((a-b).^2))
%   'manhattan'  - sum(abs(a-b))
%   'cosine'     - 1 - (a.b)/(|a||b|)
%   'p3'         - sum(abs(a-b).^3)^(1/3)

 
clear;
close all;
 
%% Load everything
train = csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train = train(:,1:784);
train(:,785) = zeros(1500,1);
 
test = csvread('mnist_test_200.csv');
correctlabels = test(:,785);
test = test(:,1:784);
test(:,785) = zeros(200,1);
 
%% numbers
k = 30;             % fix k
max_iter = 25;
num_runs = 5;     %repeat and later average
metrics = {'euclidean','manhattan','cosine','p3'};
 
% store accuracy of run: rows = runs, cols = metrics
acc_all = zeros(num_runs, length(metrics));
% keep one cost curve per metric (from its first run) for the convergence plot
cost_curves = zeros(max_iter, length(metrics));
 
%% Run each metric 5 times
for m = 1:length(metrics)
    metric = metrics{m};
 
    for r = 1:num_runs
        % train k-means (random init differs each run)
        [centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter, metric);
 
        % label centroids by majority vote
        centroid_labels = label_centroids(assignments, trainsetlabels, k);
 
        %evaluate on test set
        preds = predict_all(test, centroids, centroid_labels, metric);
 
        acc_all(r, m) = sum(preds == correctlabels) / 200;
 
        if r == 1
            cost_curves(:, m) = cost_hist;   % save first run's cost curve
        end
    end
 
    fprintf('%-12s ->  mean accuracy = %.3f  (std %.3f) over %d runs\n', ...
            metric, mean(acc_all(:,m)), std(acc_all(:,m)), num_runs);
end
 
%% useful stats
mean_acc = mean(acc_all, 1)';   % column: mean per metric
std_acc  = std(acc_all, 0, 1)'; % column: std per metric
 
%% mean accuracy with error bars
figure;
bar(mean_acc);
hold on;
errorbar(1:length(metrics), mean_acc, std_acc, 'k', 'linestyle', 'none', 'LineWidth', 1.2);
set(gca, 'XTick', 1:length(metrics), 'XTickLabel', metrics, 'FontSize', 10);
title(sprintf('Mean Test Accuracy by Distance Metric (%d runs)', num_runs), ...
      'FontSize', 16, 'FontWeight', 'bold');
xlabel('Distance Metric', 'FontSize', 14);
ylabel('Mean Test Accuracy (\pm 1 std)', 'FontSize', 14);
ylim([0 1]);
grid on;

for m = 1:length(metrics)
    text(m, mean_acc(m)+std_acc(m)+0.03, sprintf('%.3f', mean_acc(m)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end
 
%% cost convergence per metric
figure;
hold on;
colors = lines(length(metrics));
for m = 1:length(metrics)
    normalized = cost_curves(:,m) / cost_curves(1,m);
    plot(1:max_iter, normalized, 'LineWidth', 1.6, 'Color', colors(m,:));
end
title('Normalized k-means Cost Convergence by Metric', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 11);
ylabel('Cost (normalized to iteration 1)', 'FontSize', 11);
legend(metrics, 'Location', 'best');
grid on;
set(gca, 'FontSize', 10);
 
%% functions

function [centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter, metric)
    centroids = initialize_centroids(train, k);
    cost_hist = zeros(max_iter,1);
    for iter = 1:max_iter
        D = all_distances(train(:,1:784), centroids(:,1:784), metric); % 1500 x k
        [mind, idx] = min(D, [], 2);
        train(:,785)    = idx;
        cost_hist(iter) = sum(mind);
        centroids = update_centroids(train, k);
    end
    assignments = train(:,785);
end
 
function centroid_labels = label_centroids(assignments, trainsetlabels, k)
    centroid_labels = zeros(k,1);
    for j = 1:k
        assigned = trainsetlabels(assignments == j);
        if isempty(assigned)
            centroid_labels(j) = -1;
        else
            centroid_labels(j) = mode(assigned);
        end
    end
end
 
function preds = predict_all(test, centroids, centroid_labels, metric)
    D = all_distances(test(:,1:784), centroids(:,1:784), metric); % 200 x k
    [~, idx] = min(D, [], 2);
    preds = centroid_labels(idx);
end
 
function y = initialize_centroids(data, num_centroids)
    ri = randperm(size(data,1));
    y = data(ri(1:num_centroids), :);
end
 
% distance metric
%returns an (numImages x numCentroids) matrix of distances.
function D = all_distances(X, C, metric)
    switch metric
        case 'euclidean'
            xx = sum(X.^2, 2);
            cc = sum(C.^2, 2)';
            D  = sqrt(max(xx + cc - 2*(X*C'), 0));
        case 'manhattan'
            k = size(C,1);
            D = zeros(size(X,1), k);
            for j = 1:k
                D(:,j) = sum(abs(X - C(j,:)), 2);
            end
        case 'cosine'
            Xn = X ./ (sqrt(sum(X.^2,2)) + eps);
            Cn = C ./ (sqrt(sum(C.^2,2)) + eps);
            D  = 1 - Xn * Cn';
        case 'p3'
            k = size(C,1);
            D = zeros(size(X,1), k);
            for j = 1:k
                D(:,j) = sum(abs(X - C(j,:)).^3, 2).^(1/3);
            end
        otherwise
            error('Unknown metric: %s', metric);
    end
end
 
function new_centroids = update_centroids(data, K)
    new_centroids = zeros(K, size(data,2));
    for i = 1:K
        members = data(data(:,785) == i, 1:784);
        if isempty(members)
            r = randi(size(data,1));
            new_centroids(i,1:784) = data(r,1:784);
        else
            new_centroids(i,1:784) = mean(members, 1);
        end
    end
end
 
