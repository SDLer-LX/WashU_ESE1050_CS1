%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

predictions = zeros(200,1);
outliers = zeros(200,1);
%add distance for later outliers
distances = zeros(200,1);

% loop through the test set, figure out the predicted number
for i = 1:200

    testing_vector=test(i,:);

    % Extract the centroid that is closest to the test image
    [prediction_index, vec_distance]=assign_vector_to_centroid(testing_vector,centroids);

    predictions(i) = centroid_labels(prediction_index);
    distances(i) = vec_distance;

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
% FILL IN

mean_distance = mean(distances);
std_distance = std(distances);
outlier_threshold = mean_distance + 2*std_distance;

outliers(distances > outlier_threshold) = 1;
%if distance to centroid two sd above mean, flag

%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(outliers);
title('Outlier Flags','FontSize',16);
xlabel('Test Image Index','FontSize',14);
ylabel('Outlier (1) / Not Outlier (0)','FontSize',14);
ylim([-0.1 1.2]);         
yticks([0 1]);       

%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
ylim([-0.5 9.5]);          
yticks(0:9);      
title('Predictions');

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)
%percentage without outliers
sum(correctlabels(outliers==0) == predictions(outliers==0)) / sum(outliers==0)

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
num_centroids = size(centroids,1);
dists = zeros(num_centroids,1);

for k = 1:num_centroids
    dists(k) = sqrt(sum((data - centroids(k,:)).^2));
end

[vec_distance, index] = min(dists);
end
