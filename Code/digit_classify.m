function C = digit_classify(testdata) 

% A function which performs K-Nearest Neighbour (KNN) classification with 
% feature extract. Input data columns are x, y and z axis and rows
% are the observational points.

% Only a data matrix form of a single digit entered in a loop so that, 
% the third dimension of a matrix is used.

% Keep all the three file inside of data folder then continue to run the
% file. Otherwise need to add the path for the data folder before running
% the code.


% Load the trainingdata and class
load('train_profile');                  % Training data features
load('Num_data_Com');
load('N_data_Com');
load('new_Data');
load('data_Num');

traindata = train_profile;              % Training data
trainclass = Num_data_Com;
k = 3;                                  % Based on cross-validation
plot(testdata(:, 1), testdata(:, 2))


train_ratio = 0.7;                      % Training dataset ratio
test_ratio = 1 - train_ratio;           % Testing dataset ratio

id_x_Train = randsample(size(new_Data, 3), round(train_ratio* ...
    size(new_Data, 3)));                % Index of train data (random)

% Index of test data (random)
id_x_Test = setdiff(1:size(new_Data, 3), id_x_Train)';

% Train and test data
train_prof = train_profile(:, id_x_Train);
test_prof = train_profile(:, id_x_Test);

% Train and test data
train_class_prof = data_Num(id_x_Train, 1);
test_class_prof = data_Num(id_x_Test, 1);

classes = knn(train_class_prof, train_prof, test_prof, 7);

correct_class = sum(classes == test_class_prof)/size(test_class_prof, 1);

avg_correct_class = mean(correct_class);
accuracy = ((classes - (correct_class))/classes)*100

% Normalization of data
for q = 1:size(testdata, 2)
    
    testdata(:, q)=(testdata(:, q) - min(testdata(:, q)))/ ...
        (max(testdata(:, q)) - min(testdata(:, q)));
    
end


% Projections and slope information

pos = size(testdata, 1);                % Length of the stroke

% Projecion on y and x
% Define equations --> find parameter

step_size = 0.0500;                     % Since 3 decimal numbers for
                                        % the recording of the 
                                        % coordinates per axis

% Processing for the amount of projections for y
y_project = zeros(size(0:step_size:1, 2), 1);

% Processing for the amount of projections for x
x_project = zeros(size(0:step_size:1, 2), 1);


for y = 0:step_size:1
        n = 0;                          % For projection on y
        m = 0;                          % For projection on x
        for q = 1:pos - 1               % -1 since also use of q + 1
             if testdata(q, 1) < testdata(q + 1, 1)
                % If the first point is smaller than the second point, the
                % solver can find a slope
                x1 = testdata(q, 1);
                x2 = testdata(q + 1, 1);
                y1 = testdata(q, 2);
                y2 = testdata(q + 1, 2);
            else
                x2 = testdata(q, 1);
                x1 = testdata(q + 1, 1);
                y2 = testdata(q, 2);
                y1 = testdata(q + 1,2);
            end
            l = (y2 - y1)/(x2 - x1);    % For y-slope
            ll = y1 - (l*x1);            % Intersect for y
            x_val = (y - ll)/l;
            % Close and open interval
            if x_val >= min([x1, x2]) && x_val < max([x1, x2])
                n = n + 1;
            end
            y_val = l*y + ll;
            % Close and open interval
            if y_val >= min([y1, y2]) && y_val < max([y1, y2])
                m = m + 1;
                
            end
            
        end
        y_project(round((y + step_size)/step_size)) = n;
        x_project(round((y + step_size)/step_size)) = m;
end

% Train vector for the profiles
train_profile=[x_project; y_project];


% Adaptation approach

slope_val = zeros(4, 1);
w = 0.20;                               % Initial data

slope_val(1:2, 1) = [testdata(round(pos*w), 2) - testdata(1, 2);...
    testdata(round(pos*w), 1) - testdata(1, 1)];

slope_val(3, 1) = sum(diff(testdata(1:pos, 1)));
slope_val(4, 1) = sum(diff(testdata(1:pos, 2)));

data = [train_profile; slope_val];      % Features for test data


% KNN

C = zeros(1, 1);                        % Class vector based on knn
% Determine the distance
for h = 1:size(data, 2)
    dist_train = zeros(size(traindata, 2), 1);
    for j = 1:size(traindata, 2)
    dist_train(j) = (sum((traindata(:, j) - data).^2))^0.5;
    end
    dist_train_sort = sort(dist_train, 'ascend');
    k_nearest = dist_train_sort(1:1:k); % Distance of knn
    % Finding the corresponding data classes
    pos = zeros(size(k_nearest, 1), 1);
    n = 1;
    while n < k + 1
        if size(find(dist_train == k_nearest(n)), 1) == 1

            pos(n) = find(dist_train == k_nearest(n));
            n = n + 1;
            
        elseif size(find(dist_train == k_nearest(n)), 1) <= k - n + 1

            pos(n:1:n + size(find(dist_train == k_nearest(n)), 1) - 1) ...
                = find(dist_train == k_nearest(n));
             % -1 as n is also a position
            
            n = n + 1 + size(find(dist_train == k_nearest(n)), 1) - 1;
            
        else
            
            po_tv = find(dist_train == k_nearest(n));
            pos(n:1:k) = po_tv(1:1:k - n + 1); 
            n = k + 1;
            
        end
        
    end
    
    C = mode(trainclass(pos));
    % Finding the mode of k nearest neighbours.
    % If there have no single mode, then first class will be selected
  
end

end








% References
% Digital Imaging and Image Preprocessing course (DIIP), autumn, semester, 2021.
% https://www.matlabcoding.com/2020/05/building-k-nearest-neighbor-algorithm.html
% https://se.mathworks.com/help/stats/fitcknn.html
% https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
% https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
% https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning