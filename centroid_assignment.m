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

% Run k-means++ initialization and kmeans, track cost over iterations, then plot
metrics = {'kmeans++'}; % single metric for k-means++
cost_curves = zeros(max_iter, length(metrics));

% Initialize centroids with kmeans++
centroids = kmeansplusplus_initialize(train, k);

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
    centroids = new_centroids;
    % compute cost (sum of squared distances)
    sqd = sum((train - centroids(labels,:)).^2, 2);
    cost_curves(iter,1) = sum(sqd);
end

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