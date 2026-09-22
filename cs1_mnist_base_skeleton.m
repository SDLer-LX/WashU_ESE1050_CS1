
clear all;
close all;

%% In this script, you need to implement three functions as part of the k-means algorithm.
% These steps will be repeated until the algorithm converges:

  % 1. initialize_centroids
  % This function sets the initial values of the centroids
  
  % 2. assign_vector_to_centroid
  % This goes through the collection of all vectors and assigns them to
  % centroid based on norm/distance
  
  % 3. update_centroids
  % This function updates the location of the centroids based on the collection
  % of vectors (handwritten digits) that have been assigned to that centroid.


%% Initialize Data Set
% These next lines of code read in two sets of MNIST digits that will be used for training and testing respectively.

% training set (1500 images)
train=csvread('mnist_train_1500.csv');
trainsetlabels = train(:,785);
train=train(:,1:784);
train(:,785)=zeros(1500,1);

% testing set (200 images with 11 outliers)
test=csvread('mnist_test_200.csv');
% store the correct test labels
correctlabels = test(:,785);
test=test(:,1:784);

% now, zero out the labels in "test" so that you can use this to assign
% your own predictions and evaluate against "correctlabels"
% in the 'cs1_mnist_evaluate_test_set.m' script
test(:,785)=zeros(200,1);

%% After initializing, you will have the following variables in your workspace:
% 1. train (a 1500 x 785 array, containins the 1500 training images)
% 2. test (a 200 x 785 array, containing the 200 testing images)
% 3. correctlabels (a 200 x 1 array containing the correct labels (numerical
% meaning) of the 200 test images

%% To visualize an image, you need to reshape it from a 784 dimensional array into a 28 x 28 array.
% to do this, you need to use the reshape command, along with the transpose
% operation.  For example, the following lines plot the first test image

figure;
colormap('gray'); % this tells MATLAB to depict the image in grayscale
testimage = reshape(test(1,[1:784]), [28 28]);
% we are reshaping the first row of 'test', columns 1-784 (since the 785th
% column is going to be used for storing the centroid assignment.
imagesc(testimage'); % this command plots an array as an image.  Type 'help imagesc' to learn more.

%% After importing, the array 'train' consists of 1500 rows and 785 columns.
% Each row corresponds to a different handwritten digit (28 x 28 = 784)
% plus the last column, which is used to index that row (i.e., label which
% cluster it belongs to.  Initially, this last column is set to all zeros,
% since there are no clusters yet established.

%% This next section of code calls the three functions you are asked to specify

k= 60; % set k
max_iter= 100; % set the number of iterations of the algorithm
%cost_iteration graph reveals 25-ish is sufficient

%% The next line initializes the centroids.  Look at the initialize_centroids()
% function, which is specified further down this file.

centroids=initialize_centroids(train,k);

%% Initialize an array that will store k-means cost at each iteration

cost_iteration = zeros(max_iter, 1);

%% This for-loop enacts the k-means algorithm

% this loop sorts pictures to clusters and calculates cost
for iter=1:max_iter
    for i=1:(size(train,1)) %iterate over every training image
        [index, vec_distance] = assign_vector_to_centroid(train(i,:),centroids); %closest centroid
        train(i,785) = index; % which cluster this plot belongs to, put in tag on 785 column
        %do square if euclidean
        cost_iteration(iter) = cost_iteration(iter) + vec_distance^2;
    end

    centroids = update_Centroids(train,k); %use the update centroid function
end
%% Determine the label of each centroid based on majority vote of assigned training images
centroid_labels = zeros(k,1);

for j = 1:k
    assigned_labels = trainsetlabels(train(:,785) == j);

    if isempty(assigned_labels)
        centroid_labels(j) = -1;
    else
        centroid_labels(j) = mode(assigned_labels);
    end
end

%% Save the trained classifier for submission
save('classifierdata.mat', 'centroids', 'centroid_labels');
%% This section of code plots the k-means cost as a function of the number
% of iterations

figure;

plot(cost_iteration,"LineWidth",1.5);
xlabel('Iteration');
ylabel('K-means Cost');
title('K-means Cost vs Iteration'); %consider changes for readability like line-width etc.


%% This next section of code will make a plot of all of the centroids
% Again, use help <functionname> to learn about the different functions
% that are being used here.

figure;
colormap('gray');

plotsize = ceil(sqrt(k));

for ind=1:k
    
    centr=centroids(ind,[1:784]);
    subplot(plotsize,plotsize,ind);
    
    imagesc(reshape(centr,[28 28])');
    title(strcat('Centroid ',num2str(ind)));

end

%% Function to initialize the centroids
% This function randomly chooses k vectors from our training set and uses them to be our initial centroids
% There are other ways you might initialize centroids.
% ***Feel free to experiment.***
% Note that this function takes two inputs and emits one output (y).

%input: whole training set, k
%output: a set of k random centroids
function y=initialize_centroids(data,num_centroids)

random_index=randperm(size(data,1)); %shuffle to random order

centroids=data(random_index(1:num_centroids),:); % get the first k rows

y=centroids;

end

%% Function to pick the Closest Centroid using norm/distance
% This function takes two arguments, a vector and a set of centroids
% It returns the index of the assigned centroid and the distance between
% the vector and the assigned centroid.

%input: one picture, centroid set
%output: which centroid this picture is closest to
function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
  %numbers needed
  num_centroids = size(centroids,1); %number of centroids
  distance = zeros(num_centroids,1); %empty vector of zeros for storing distance

  %nearest neighbor calculation
  for i = 1:num_centroids
      % calculate pixel-pixel difference, square them, and add them 
      distance(i) = sqrt(sum ((data(1:784)- centroids(i,1:784)).^2)); 
  end
  %all of the above can be accomplished via the vecnorm function --->
  %vecnorm(centroids(:,1:784) - data(1:784), 2, 2); 
  %but this implementation seems just as well

  [vec_distance,index] = min(distance);
end


%% Function to compute new centroids using the mean of the vectors currently assigned to the centroid.
% This function takes the set of training images and the value of k.
% It returns a new set of centroids based on the current assignment of the
% training images.

function new_centroids=update_Centroids(data,K)

new_centroids = zeros(K, size(data,2)); % create empty K x 785 vector, size has parameter 2 so that it doesnt return both column and row.
%i'm keeping with the convention that our parameters are generalizable and not specific to the 'train' dataset

for i=1:K %do once for each centroid
    members = data(data(:,785)==i, 1:784); %find every entry belonging to that centroid
    if isempty(members)
        new_centroids(i,1:784) = data(randi(size(data,1)),1:784); % reseed empty cluster
    else
        new_centroids(i,1:784) = mean(members,1);
    end
    %set new centroid to mean of all of its members
end

end