clc
clear all
close all

% Data loading
input_data = zeros(100, 3, 1000);          % According to input data to
                                           % Createing 100 points for x,y,z
size_Of_Data = [100 100 100 100 100 100 100 100 100 100];

% Class vectors
data_Num = [repmat(0,size_Of_Data(1), 1); repmat(1, size_Of_Data(2), ...
    1); repmat(2, size_Of_Data(3), 1); repmat(3, size_Of_Data(4), 1); ...
    repmat(4, size_Of_Data(5), 1); repmat(5, size_Of_Data(6), 1); ...
    repmat(6, size_Of_Data(7), 1); repmat(7, size_Of_Data(8), 1); ...
    repmat(8, size_Of_Data(9), 1); repmat(9, size_Of_Data(10), 1)];


size_Of_Data_Cum = cumsum(size_Of_Data);
size_Of_Data_Cum = [0 size_Of_Data_Cum];   % Zero for the loop (later)

stroke_Size = [];                          % Rows


% Loop for extracting data
for i = 1:10                               % For 0 to 9
        for j = 1:size_Of_Data(i)
            if j < 10
            C = strcat('stroke_', num2str(i-1), '_000', ...
                num2str(j), '.mat');
            elseif (j >= 10 && j < 100)
                C = strcat('stroke_', num2str(i-1), '_00', ...
                    num2str(j), '.mat');
                else C = strcat('stroke_', num2str(i-1), '_0', ...
                        num2str(j), '.mat');
            end
            load(C);
            % Until 1000 strokes
            stroke_Size = [stroke_Size, size(pos, 1)];
            
            for k = 1:size(pos, 2)
            pos(:, k) = (pos(:, k) - min(pos(:, k)))/(max(pos(:, k)) ...
                -min(pos(:, k)));          % Normalization using min-max
            
            end
            
            input_data(1:size(pos, 1), :, size_Of_Data_Cum(i) + j) = pos;
            
        end
end

new_Data = input_data;

save('new_Data.mat', 'new_Data')
save('stroke_Size.mat', 'stroke_Size')
save('data_Num.mat', 'data_Num')


% Rotation on the x-axis

% Angle of rotation
deg_Of_Rot = [5, 10, 15, 20, 25, 30, 35, 40, 45];

% Processing
n_Rot_X_Data_Set = zeros(100, 3, size(input_data, 3)*size(deg_Of_Rot, 2));

stroke_Size_X_Rot = repmat(stroke_Size, 1, size(deg_Of_Rot, 2));

for i = 1:size(deg_Of_Rot, 2)              % Size of rotation vector
for j = 1:size(input_data, 3)              % Size of data (3rd dim)
angle = deg2rad(deg_Of_Rot(i));            % Changing degree to radian


% Rotation around x axis (Using DIIP)
R_3D_Rot = [cos(angle) 0 sin(angle); 0 1 0; -sin(angle) 0 ...
    cos(angle)];

data_Set_Of_Rot = input_data(1:stroke_Size(j), :, j)*R_3D_Rot;
n_Rot_X_Data_Set(1:size(data_Set_Of_Rot, 1), :, (i - 1)* ...
    size(input_data, 3) + j) = bsxfun(@minus, data_Set_Of_Rot, ...
    min(data_Set_Of_Rot, [], 1))./repmat((max(data_Set_Of_Rot, ...
    [], 1) - min(data_Set_Of_Rot, [], 1)), size(data_Set_Of_Rot, 1), 1);
% Each step of j a new matrix adapted from the initial matrix
% in 3rd dimension of the data_rotation matrix

end
end

Num_data_x_rot = repmat(data_Num, size(deg_Of_Rot, 2), 1);

save('n_Rot_X_Data_Set.mat', 'n_Rot_X_Data_Set')
save('stroke_Size_X_Rot.mat', 'stroke_Size_X_Rot')
save('Num_data_x_rot.mat', 'Num_data_x_rot')


% Rotation on y-axis

% Processing

% Angle of rotation
deg_Of_Rot = [5, 10, 15, 20, 25, 30, 35, 40, 45];

n_Rot_Y_Data_Set = zeros(100, 3, size(input_data, 3)* ...
    size(deg_Of_Rot, 2));

stroke_Size_Y_Rot = repmat(stroke_Size, 1, size(deg_Of_Rot, 2));

for i = 1:size(deg_Of_Rot, 2)              % Size of rotation vector
for j = 1:size(input_data, 3)              % Size of data (3rd dim)
    
angle = deg2rad(deg_Of_Rot(i));            % Changing degree to radian

% Rotation around y axis (Using DIIP)
R_3D_Rot = [cos(angle) 0 -sin(angle); 0 1 0; sin(angle) 0 ...
    cos(angle)];

data_Set_Of_Rot = input_data(1:stroke_Size(j), :, j)*R_3D_Rot;

n_Rot_Y_Data_Set(1:size(data_Set_Of_Rot, 1), :, (i - 1)* ...
    size(input_data, 3) + j) = bsxfun(@minus, data_Set_Of_Rot, ...
    min(data_Set_Of_Rot, [], 1))./repmat((max(data_Set_Of_Rot, ...
    [], 1) - min(data_Set_Of_Rot, [], 1)), size(data_Set_Of_Rot, 1), 1);
% Each step of j a new matrix adapted from the initial matrix
% in 3rd dimension of the data_rotation matrix

end
end

Num_data_y_rot = repmat(data_Num, size(deg_Of_Rot, 2), 1);

save('n_Rot_Y_Data_Set.mat', 'n_Rot_Y_Data_Set') 
save('stroke_Size_Y_Rot.mat', 'stroke_Size_Y_Rot') 
save('Num_data_y_rot.mat', 'Num_data_y_rot')

% Rotation on z-axis

% Processing

% Angle of rotation
deg_Of_Rot = [5, 10, 15, 20, 25, 30, 35, 40, 45];

n_Rot_Z_Data_Set = zeros(100, 3, size(input_data, 3)*size(deg_Of_Rot, 2));

stroke_Size_Z_Rot = repmat(stroke_Size, 1, size(deg_Of_Rot, 2));

for i = 1:size(deg_Of_Rot, 2)              % Size of rotation vector
for j = 1:size(input_data, 3)              % Size of data (3rd dim)

angle = deg2rad(deg_Of_Rot(i));            % Changing degree to radian

% Rotation around z axis (Using DIIP)
R_3D_Rot =[cos(angle) sin(angle) 0; -sin(angle) cos(angle) 0; ...
    0 0 1];

data_Set_Of_Rot = input_data(1:stroke_Size(j), :, j)*R_3D_Rot;

n_Rot_Z_Data_Set(1:size(data_Set_Of_Rot, 1), :, (i - 1)* ...
    size(input_data, 3) + j) = bsxfun(@minus, data_Set_Of_Rot, ...
    min(data_Set_Of_Rot, [], 1))./repmat((max(data_Set_Of_Rot, [], 1) ...
    - min(data_Set_Of_Rot, [], 1)), size(data_Set_Of_Rot, 1), 1);
% Each step of j a new matrix adapted from the initial matrix
% in 3rd dimension of the data_rotation matrix

end
end

Num_data_z_rot = repmat(data_Num, size(deg_Of_Rot, 2), 1);

save('n_Rot_Z_Data_Set.mat', 'n_Rot_Z_Data_Set') 
save('stroke_Size_Z_Rot.mat', 'stroke_Size_Z_Rot') 
save('Num_data_z_rot.mat', 'Num_data_z_rot')


% Combining rotated x, y, z axis data
size_data = size(new_Data, 3) + size(n_Rot_X_Data_Set, 3) + ...
    size(n_Rot_Y_Data_Set, 3) + size(n_Rot_Z_Data_Set, 3);

N_data_Com = zeros(222, 3, size_data);
N_data_Com(:, :, 1: size(new_Data, 3)) = new_Data;

N_data_Com(:, :, size(new_Data, 3) + 1:size(new_Data, 3) + ...
    size(n_Rot_X_Data_Set, 3)) = n_Rot_X_Data_Set;

N_data_Com(:, :, size(new_Data, 3) + size(n_Rot_X_Data_Set, 3) + ...
    1:size(new_Data, 3) + size(n_Rot_X_Data_Set, 3) + ...
    size(n_Rot_Y_Data_Set, 3)) = n_Rot_Y_Data_Set;

N_data_Com(:, :, size(new_Data, 3) + size(n_Rot_X_Data_Set, 3) + ...
    size(n_Rot_Y_Data_Set, 3) + 1:size_data) = n_Rot_Z_Data_Set;

stroke_Size_Com = [stroke_Size stroke_Size_X_Rot stroke_Size_Y_Rot ...
    stroke_Size_Z_Rot];

% Combining classes
Num_data_Com = [data_Num; Num_data_x_rot; Num_data_y_rot; ...
    Num_data_z_rot];

save('N_data_Com.mat', 'N_data_Com') 
save('stroke_Size_Com.mat', 'stroke_Size_Com') 
save('Num_data_Com.mat', 'Num_data_Com')


% Adaptation approach
slope_val = zeros(4, size(stroke_Size_Com, 2));
w = 0.20;                                  % Initial data

for m = 1:size(stroke_Size_Com, 2)
    
    slope_val(1:2, m) = [N_data_Com(round(stroke_Size_Com(m)*w), 2, m) ...
        - N_data_Com(1, 2, m); N_data_Com(round(stroke_Size_Com(m)*w), ...
        1, m) - N_data_Com(1, 1, m)];
    
    slope_val(3, m) = sum(diff(N_data_Com(1:stroke_Size_Com(m), 1, m)));
    slope_val(4, m) = sum(diff(N_data_Com(1:stroke_Size_Com(m), 2, m)));
    
end


% Classify

% Loading data
load('N_data_Com.mat')
load('stroke_Size_Com.mat')
load('Num_data_Com.mat')                   % Class data


% Projecion on y and x

step_size = 0.0500;                        % Since 3 decimal numbers for
                                           % the recording of the 
                                           % coordinates per axis

% Processing for the amount of projections for y
project_y = zeros(size(0:step_size:1, 2), 1);

% Processing for the amount of projections for x
project_x = zeros(size(0:step_size:1, 2), 1);

% Processing of projections for all observations
profils_num_y = zeros(size(0:step_size:1, 2), size(N_data_Com, 3));
profils_num_x = zeros(size(0:step_size:1, 2), size(N_data_Com, 3));


for k = 1:size(N_data_Com, 3) 
    for y = 0:step_size:1
        n = 0;                             % For projection on y
        m = 0;                             % For projection on x
        for q = 1:stroke_Size_Com(k) - 1   % -1 since also use of q + 1
             if N_data_Com(q, 1, k) < N_data_Com(q + 1, 1, k)
                % If the first point is smaller than the second point, the
                % solver can find a slope
                x1 = N_data_Com(q, 1, k);
                y1 = N_data_Com(q, 2, k);
                x2 = N_data_Com(q + 1, 1, k);
                y2 = N_data_Com(q + 1, 2, k);
            else
                x2 = N_data_Com(q, 1, k);
                y2 = N_data_Com(q, 2, k);
                x1 = N_data_Com(q + 1, 1, k);
                y1 = N_data_Com(q + 1, 2, k);
            end
            l = (y2 - y1)/(x2 - x1);       % For y-slope
            ll = y1 - (l*x1);              % Intersect for y
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
        
        project_y(round((y + step_size)/step_size)) = n;
        project_x(round((y + step_size)/step_size)) = m;
    
    end
    
    profils_num_y(:, k) = project_y;
    profils_num_x(:, k) = project_x;
end

% Train vector for the profiles
train_profile = [profils_num_x; profils_num_y];
train_profile = [train_profile; slope_val];

save('train_profile.mat', 'train_profile')
load('data_Num.mat')                       % Class data


% Adaptation of train class 0-9 --> 1-10

train_rate = 0.7;                         % Size of train dataset

x_Train = randsample(size(new_Data, 3), round(train_rate* ...
    size(new_Data, 3)));                   % Random index of train data

% Random index of test data
x_Test = setdiff(1:size(new_Data, 3), x_Train)';

% Train and Test data
train_pro = train_profile(:, x_Train);
train_classes = data_Num(x_Train, 1);
test_pro = train_profile(:, x_Test);
test_classes = data_Num(x_Test, 1);

% Fitting knn
classes = knn(train_classes, train_pro, test_pro, 7);

% Measurements
correct_classes = sum(classes == test_classes)/ ...
    size(test_classes, 1);

avg_correct_class = mean(correct_classes);

accuracy = ((classes - (correct_classes))/classes)*100


% Testing of a single digit data
load('stroke_0_0099.mat')
x1 = pos(:, 1);
y1 = pos(:, 2);
plot(x1, y1)
Clas = digit_classify(pos)












% References
% Digital Imaging and Image Preprocessing course (DIIP), autumn, semester, 2021.
% https://www.matlabcoding.com/2020/05/building-k-nearest-neighbor-algorithm.html
% https://se.mathworks.com/help/stats/fitcknn.html
% https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
% https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
% https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning