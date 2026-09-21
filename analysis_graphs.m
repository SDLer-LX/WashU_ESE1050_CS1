
clear;
close all;

%% Load stuff
train = csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train = train(:,1:784);
train(:,785) = zeros(1500,1);

test = csvread('mnist_test_200.csv');
correctlabels = test(:,785);
test = test(:,1:784);
test(:,785) = zeros(200,1);

max_iter = 50;   % k-means iterations (cost curve shows it converges well before this)

%% try k values
k_values = [10 20 30 50 75 100 110 120];
accuracy = zeros(length(k_values),1);
final_cost = zeros(length(k_values),1);

for ki = 1:length(k_values)
    k = k_values(ki);

    % train k-means
    [centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter);

    % label each centroid by majority vote of its training images
    centroid_labels = label_centroids(assignments, trainsetlabels, k);

    % evaluate on the test set
    [preds, ~] = predict_all(test, centroids, centroid_labels);

    accuracy(ki)  = sum(preds == correctlabels) / 200;
    final_cost(ki) = cost_hist(end);

    fprintf('k = %3d  ->  accuracy = %.3f\n', k, accuracy(ki));
end

%% Plot: Accuracy vs k  (the headline result)
figure;
plot(k_values, accuracy, '-o', 'LineWidth', 1.6, 'MarkerSize', 7, ...
     'MarkerFaceColor', [0 0.45 0.74]);
title('Test Accuracy vs. Number of Centroids (k)', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('k (number of centroids)', 'FontSize', 11);
ylabel('Test Accuracy', 'FontSize', 11);
ylim([0 1]);
grid on;
set(gca, 'FontSize', 10);

%% Plot: Final cost vs k
figure;
plot(k_values, final_cost, '-s', 'LineWidth', 1.6, 'MarkerSize', 7, ...
     'MarkerFaceColor', [0.85 0.33 0.1], 'Color', [0.85 0.33 0.1]);
title('Final k-means Cost vs. k', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('k (number of centroids)', 'FontSize', 11);
ylabel('Final k-means Cost', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 10);

%% One chosen k
k = 30;
[centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter);
centroid_labels = label_centroids(assignments, trainsetlabels, k);
[predictions, distances] = predict_all(test, centroids, centroid_labels);

%% Plot: cost convergence for this k
figure;
plot(1:max_iter, cost_hist, 'LineWidth', 1.6);
title(sprintf('k-means Cost vs. Iteration (k = %d)', k), 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Iteration', 'FontSize', 11);
ylabel('k-means Cost', 'FontSize', 11);
grid on;
set(gca, 'FontSize', 10);

%% Plot: the learned centroids as images
figure;
colormap('gray');
plotsize = ceil(sqrt(k));
for ind = 1:k
    subplot(plotsize, plotsize, ind);
    imagesc(reshape(centroids(ind,1:784), [28 28])');
    title(num2str(centroid_labels(ind)));
    axis off;
end
sgtitle(sprintf('Learned Centroids (k = %d), titled by voted digit', k));

%% Outlier detection + stem plot
mean_d = mean(distances);
std_d  = std(distances);
threshold = mean_d + 2*std_d;
outliers = double(distances > threshold);

figure;
stem(outliers, 'filled', 'LineWidth', 1.2, 'Color', [0.85 0.33 0.1], ...
     'MarkerFaceColor', [0.85 0.33 0.1], 'MarkerSize', 5);
title('Outlier Flags', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Test Image Index', 'FontSize', 11);
ylabel('Outlier Flag', 'FontSize', 11);
ylim([-0.1 1.2]);
yticks([0 1]);
yticklabels({'Normal','Outlier'});
grid on;
set(gca, 'FontSize', 10);

%% Accuracy excluding outliers (cleaner number)
clean_acc = sum(correctlabels(outliers==0) == predictions(outliers==0)) / sum(outliers==0);
fprintf('\nk = %d:\n', k);
fprintf('  accuracy including outliers = %.3f\n', sum(predictions==correctlabels)/200);
fprintf('  accuracy excluding outliers = %.3f\n', clean_acc);

%% Plot: predictions vs correct labels
figure;
plot(correctlabels, 'o', 'MarkerSize', 7, 'LineWidth', 1.2);
hold on;
plot(predictions, 'x', 'MarkerSize', 8, 'LineWidth', 1.2);
title('Predictions vs. Correct Labels', 'FontSize', 13, 'FontWeight', 'bold');
xlabel('Test Image Index', 'FontSize', 11);
ylabel('Digit Label', 'FontSize', 11);
ylim([-0.5 9.5]);
yticks(0:9);
legend('Correct', 'Predicted', 'Location', 'best');
grid on;
set(gca, 'FontSize', 10);

%% Plot: some misclassified test images (confusion examples)
wrong_idx = find(predictions ~= correctlabels & outliers == 0); % wrong, non-outlier
num_show = min(9, length(wrong_idx));
if num_show > 0
    figure;
    colormap('gray');
    for m = 1:num_show
        idx = wrong_idx(m);
        subplot(3,3,m);
        imagesc(reshape(test(idx,1:784), [28 28])');
        title(sprintf('true %d, pred %d', correctlabels(idx), predictions(idx)));
        axis off;
    end
    sgtitle('Examples of Misclassified Digits');
end

%% ================================================================
%  LOCAL FUNCTIONS
%  ================================================================

function [centroids, cost_hist, assignments] = run_kmeans(train, k, max_iter)
    centroids = initialize_centroids(train, k);
    cost_hist = zeros(max_iter,1);
    for iter = 1:max_iter
        for i = 1:size(train,1)
            [index, vec_distance] = assign_vector_to_centroid(train(i,:), centroids);
            train(i,785) = index;
            cost_hist(iter) = cost_hist(iter) + vec_distance;
        end
        centroids = update_centroids(train, k);
    end
    assignments = train(:,785);   % return final cluster assignment for each training image
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

function [preds, dists] = predict_all(test, centroids, centroid_labels)
    n = size(test,1);
    preds = zeros(n,1);
    dists = zeros(n,1);
    for i = 1:n
        [idx, d] = assign_vector_to_centroid(test(i,:), centroids);
        preds(i) = centroid_labels(idx);
        dists(i) = d;
    end
end

function y=initialize_centroids(data,num_centroids)

random_index=randperm(size(data,1)); %shuffle to random order

centroids=data(random_index(1:num_centroids),:); % get the first k rows

y=centroids;

end

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
%numbers needed
num_centroids = size(centroids,1); %number of centroids
distance = zeros(num_centroids,1); %empty vector of zeros for storing distance

%nearest neighbor calculation
for i = 1:num_centroids
    % calculate pixel-pixel difference, square them, and add them up
    distance(i) = 1 - dot(data(1:784), centroids(i,1:784)) / (norm(data(1:784)) * norm(centroids(i,1:784)) + eps);
    %distance(i) = sqrt(sum ((data(1:784)- centroids(i,1:784)).^2)); 
end
%all of the above can be accomplished via the vecnorm function --->
%vecnorm(centroids(:,1:784) - data(1:784), 2, 2); 
%but this implementation seems just as well

[vec_distance,index] = min(distance);
end

function new_centroids=update_centroids(data,K)

new_centroids = zeros(K, size(data,2)); % create empty K x 785 vector, size has parameter 2 so that it doesnt return both column and row.
%i'm keeping with the convention that our parameters are generalizable and not specific to the 'train' dataset

for i=1:K %do once for each centroid
    members = data(data(:,785)==i, 1:784); %find every entry belonging to that centroid
    c = mean(members,1);
    new_centroids(i,1:784) = c / (norm(c) + eps);  %set new centroid to mean of all of its members
end

end