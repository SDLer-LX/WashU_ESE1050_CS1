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

%% Numbers

k = 30;
max_iter = 25;
num_runs = 100;

%store accuracy of run: rows = runs, cols = kmeans vs kmeans++
acc_all = zeros(num_runs, 2);

% keep one cost curve per metric (from its first run) for the convergence plot
cost_curves = zeros(max_iter, 2);

% Initialize centroids with kmeans regularly



for r = 1:num_runs
    % Initialize centroids randomly for this run
    centroids = initialize_centroids(train, k);
    labels = zeros(size(train, 1), 1);
    % train k-means (random init differs each run)
    [centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter, 'euclidean', centroids);

    % label centroids by majority vote
    centroid_labels = label_centroids(assignments, trainsetlabels, k);

    %evaluate on test set
    preds = predict_all(test, centroids, centroid_labels, 'euclidean');

    acc_all(r, 1) = sum(preds == correctlabels) / 200;

    if r == 1
        cost_curves(:, 1) = cost_hist;   % save first run's cost curve
    end
end

fprintf('%-12s ->  mean accuracy = %.3f  (std %.3f) over %d runs\n', ...
    'k-means', mean(acc_all(:,1)), std(acc_all(:,1)), num_runs);



for r = 1:num_runs
    centroids = kmeansplusplus_initialize(train, k);
    labels = zeros(size(train,1),1);
    % train k-means (random init differs each run)
    [centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter, 'euclidean', centroids);
 
    % label centroids by majority vote
    centroid_labels = label_centroids(assignments, trainsetlabels, k);
 
    %evaluate on test set
    preds = predict_all(test, centroids, centroid_labels, 'euclidean');
 
    acc_all(r, 2) = sum(preds == correctlabels) / 200;
 
    if r == 1
        cost_curves(:, 2) = cost_hist;   % save first run's cost curve
    end
end
 
    fprintf('%-12s ->  mean accuracy = %.3f  (std %.3f) over %d runs\n', ...
            'k-means++', mean(acc_all(:,2)), std(acc_all(:,2)), num_runs);



%% Implementation

function y=kmeansplusplus_initialize(data,num_centroids)

random_index=randperm(size(data,1)); %shuffle to random order

centroids = data(random_index(1:num_centroids), :); %first centroid initialized is random

for centroidIndex = 2:num_centroids
    distances = min(pdist2(data, centroids(1:centroidIndex-1, :)).^2, [], 2); %compute squared distance to all previously assigned centroids
    probabilities = distances / sum(distances);
    centroids(centroidIndex, :) = data(randsample(size(data, 1), 1, true, probabilities), :);
    %produce centroid with weighted probability based on distance
end

y=centroids;

end


function [centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter, metric, centroids)
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

function y = initialize_centroids(data, num_centroids)
    ri = randperm(size(data,1));
    y = data(ri(1:num_centroids), :);
end

function D = all_distances(X, C, metric)
switch metric
    case 'euclidean'
        xx = sum(X.^2, 2);
        cc = sum(C.^2, 2)';
        D  = sqrt(max(xx + cc - 2*(X*C'), 0));

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

% Bad old code below if still needed

%{

% Run k-means and record objective after each iteration
labels = zeros(size(train,1),1);
for iter = 1:max_iter
    % assign points
    D = pdist2(train, centroids);
    [~,labels] = min(D, [], 2);
    % recompute centroids
    new_centroids = zeros(size(centroids));
    for j = 1:k
        pts = train(labels==j, :);
        if isempty(pts)
            new_centroids(j,:) = centroids(j,:); % keep if empty
        else
            new_centroids(j,:) = mean(pts,1);
        end
    end
end

%}