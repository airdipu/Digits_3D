clc
clear all
close all

% Data loading
data_in = zeros(100, 3, 1000);             % According to input data to
                                           % create 100 points for each axis

data_size = [100 100 100 100 100 100 100 100 100 100];

% Class vectors
data_class = [repmat(0, data_size(1), 1); repmat(1, data_size(2), 1); ...
    repmat(2, data_size(3), 1); repmat(3, data_size(4), 1); ...
    repmat(4, data_size(5), 1); repmat(5, data_size(6), 1); ...
    repmat(6, data_size(7), 1); repmat(7, data_size(8), 1); ...
    repmat(8, data_size(9), 1); repmat(9, data_size(10), 1)];

data_size_cum = cumsum(data_size);
data_size_cum = [0 data_size_cum];         % Zero for the loop (later)
stroke_size = [];                          % rows

% Loop for extracting data
for i = 1:10                               % Class 0 to 9
    for j=1:data_size(i)
        if j<10
            d = strcat('stroke_', num2str(i-1), '_000', num2str(j), '.mat');
        elseif(j>=10 && j<100)
            d = strcat('stroke_', num2str(i-1), '_00', num2str(j), '.mat');
        else d = strcat('stroke_', num2str(i-1), '_0', num2str(j), '.mat');
        end
        
        load(d);
        % Until 1000 strokes
        stroke_size = [stroke_size, size(pos, 1)];
        for k = 1:size(pos, 2)
            pos(:, k) = (pos(:, k) - min(pos(:, k)))/(max(pos(:, k)) - ...
                min(pos(:, k)));           % Normalization using min-max
        end
        
            data_in(1:size(pos, 1), :, data_size_cum(i) + j) = pos;
    end
    
end

data = data_in;
save ('data.mat','data')
save('stroke_size.mat', 'stroke_size')
save('data_class.mat','data_class')



% Rotation on x-axis

% Angle of rotation
rotation_angle = [5, 10, 15, 20, 25, 30, 35, 40, 45];

% Processing
n_rotation_x_data = zeros(100, 3, size(data_in, 3)* ...
    size(rotation_angle, 2));

stroke_size_x_rotation = repmat(stroke_size, 1, size(rotation_angle, 2));

for i=1:size(rotation_angle, 2)             % Size of rotation vector
    for j=1:size(data_in, 3)                % Size of data (3rd dim)
        angle = deg2rad(rotation_angle(i)); % Changing degree to radian

        % Rotation around x axis (Using DIIP)
        rotate_around_x = [cos(angle) 0 sin(angle); 0 1 0; - sin(angle) ...
            0 cos(angle)];

        data_rotation_x = data_in(1:stroke_size(j), :, j)* ...
            rotate_around_x;
        
        n_rotation_x_data(1:size(data_rotation_x, 1), :, ...
            (i-1)*size(data_in, 3) + j) = bsxfun(@minus, ...
            data_rotation_x, min(data_rotation_x, [], 1))...
            ./repmat((max(data_rotation_x, [] , 1) - ...
            min(data_rotation_x, [], 1)), size(data_rotation_x, 1), 1);
        
        % Each step of j a new matrix adapted from the initial matrix
        % in 3rd dimension of the data_rotation matrix
    end
end

n_rotation_x_data = repmat(data_class, size(rotation_angle, 2), 1);

save ('n_rotation_x_data.mat', 'n_rotation_x_data')
save ('stroke_size_x_rotation.mat', 'stroke_size_x_rotation')
save ('n_rotation_x_data.mat', 'n_rotation_x_data')



% Rotation on y-axis

% Processing
n_rotation_y_data = zeros(100, 3, size(data_in, 3)* ...
    size(rotation_angle, 2));

stroke_size_y_rotation = repmat(stroke_size, 1, size(rotation_angle, 2));

for i=1:size(rotation_angle, 2)             % Size of rotation vector
    for j=1:size(data_in, 3)                % Size of data (3rd dim)
        angle = deg2rad(rotation_angle(i)); % Changing degree to radian

        % Rotation around y axis (Using DIIP)
        rotate_around_y = [cos(angle) 0 -sin(angle); 0 1 0; sin(angle) ...
            0 cos(angle)];

        data_rotation_y = data_in(1:stroke_size(j), :, j)* ...
            rotate_around_y;
        
        n_rotation_y_data(1:size(data_rotation_y, 1), :, ...
            (i-1)*size(data_in, 3) + j) = bsxfun(@minus, ...
            data_rotation_y, min(data_rotation_y, [], 1))...
            ./repmat((max(data_rotation_y, [] , 1) - ...
            min(data_rotation_y, [], 1)), size(data_rotation_y, 1), 1);
        
        % Each step of j a new matrix adapted from the initial matrix
        % in 3rd dimension of the data_rotation matrix
    end
end

n_rotation_y_data = repmat(data_class, size(rotation_angle, 2), 1);

save ('n_rotation_y_data.mat', 'n_rotation_y_data')
save ('stroke_size_y_rotation.mat', 'stroke_size_y_rotation')
save ('n_rotation_y_data.mat', 'n_rotation_y_data')



% Rotation on z-axis

% Processing
n_rotation_z_data = zeros(100, 3, size(data_in, 3)* ...
    size(rotation_angle, 2));

stroke_size_z_rotation = repmat(stroke_size, 1, size(rotation_angle, 2));

for i=1:size(rotation_angle, 2)             % Size of rotation vector
    for j=1:size(data_in, 3)                % Size of data (3rd dim)
        angle = deg2rad(rotation_angle(i)); % Changing degree to radian

        % Rotation around z axis (Using DIIP)
        rotate_around_z = [cos(angle) sin(angle) 0; -sin(angle) ...
            cos(angle) 0; 0 0 1];

        data_rotation_z = data_in(1:stroke_size(j), :, j)* ...
            rotate_around_y;
        
        n_rotation_z_data(1:size(data_rotation_z, 1), :, ...
            (i-1)*size(data_in, 3) + j) = bsxfun(@minus, ...
            data_rotation_z, min(data_rotation_z, [], 1))...
            ./repmat((max(data_rotation_z, [] , 1) - ...
            min(data_rotation_z, [], 1)), size(data_rotation_z, 1), 1);
        
        % Each step of j a new matrix adapted from the initial matrix
        % in 3rd dimension of the data_rotation matrix
    end
end

n_rotation_z_data = repmat(data_class, size(rotation_angle, 2), 1);

save ('n_rotation_z_data.mat', 'n_rotation_z_data')
save ('stroke_size_z_rotation.mat', 'stroke_size_z_rotation')
save ('n_rotation_z_data.mat', 'n_rotation_z_data')





% Combining rotated x, y, z axis data
size_data = size(data, 3) + size(n_rotation_x_data, 3) + ...
    size(n_rotation_x_data, 3) + size(n_rotation_x_data, 3);

data_combine = zeros(106, 3, size_data);
data = data_combine(:, :, 1:size(data, 3));

n_rotation_x_data = data_combine(:, :, size(data, 3) + 1:size(data, 3) ...
    + size(n_rotation_x_data, 3));

n_rotation_y_data = data_combine(:, :, size(data, 3) + ...
    size(n_rotation_x_data, 3) + 1:size(data, 3) + ...
    size(n_rotation_y_data, 3) + size(n_rotation_y_data, 3));

n_rotation_z_data = data_combine(:,:,size(data,3) + ...
    size(n_rotation_x_data, 3) + size(n_rotation_y_data, 3) + ...
    1:size_data);

stroke_size_combine = [stroke_size stroke_size_x_rotation ...
    stroke_size_y_rotation stroke_size_z_rotation];

% Combining classes
n_class_combine = [data_class; n_rotation_x_data; n_rotation_y_data; ...
    n_rotation_z_data];

save ('data_combine.mat', 'data_combine') 
save ('stroke_size_combine.mat', 'stroke_size_combine') 
save ('n_data_combine.mat', 'n_data_combine')



% Adpation 10% Approach
slope = zeros(4, size(stroke_size_combine, 2));
p = 0.20;     % 10% of initial data

for l = 1:size(stroke_size_combine, 2)
    slope(1:2, l) = [data_combine(round(stroke_size_combine(l)*p), 2, ...
        m) - data_combine(1, 2, m); data_combine(round( ...
        stroke_size_combine(l)*p), 1, l) - data_combine(1, 1, l)];
    
    slope(3, l) = sum(diff(data_combine(1:stroke_size_combine(l), 1, l)));
    slope(4, l) = sum(diff(data_combine(1:stroke_size_combine(l), 2, l)));
end


% Classify

% Loading data
load('data_combine.mat')
load('stroke_size_combine.mat')
load('n_class_combine.mat')


% Projecion on y and x
% define equations --> find parameter

step_size = 0.0500;                         % Since 3 decimal numbers for
                                            % the recording of the 
                                            % coordinates per axis
% Processing for the amount of projections
projection_y = zeros(size(0:step_size:1, 2), 1);

% Processing for the amount of projections
projection_x = zeros(size(0:step_size:1, 2), 1);

% Processing of projections for all observations
profil_y = zeros(size(0:step_size:1, 2), size(data_combine, 3));
profil_x = zeros(size(0:step_size:1, 2), size(data_combine, 3));

for k = 1:size(data_combine, 3)
    for y = 0:step_size:1
        l1 = 0;                             % For projection on y
        l2 = 0;                             % For projection on x
        for q = 1:stroke_size_combine(k) - 1% -1 since also use of q+1
            % If the first point is smaller than the second point, the
            % solver can find a slope
            if data_combine(q, 1, k) < data_combine(q+1, 1, k)
                x1 = data_combine(q, 1, k);
                y1 = data_combine(q, 2, k);
                x2 = data_combine(q+1, 1, k);
                y2 = data_combine(q+1, 2, k);
            else
                x2 = data_combine(q, 1, k);
                y2 = data_combine(q, 2, k);
                x1 = data_combine(q+1, 1, k);
                y1 = data_combine(q+1, 2, k);
            end
            
            a = (y2 - y1)/(x2 - x1);        % For y-slope
            b = y1 - (a*x1);                % Intersect for y
            x_val = (y - b)/a;
            % Close and open interval
            if x_val >= min([x1, x2]) && x_val < max([x1, x2])
                l1 = l1 + 1;
            end
            y_val = a*y + b;
            % Close and open interval
            if y_val >= min([y1, y2]) && y_val < max([y1, y2])
                l2 = l2 + 1;
            end
        end
        l1 = projection_y(round((y + step_size)/step_size));
        l2 = projection_x(round((y + step_size)/step_size));
    end
    project_y = profil_y(:, k);
    project_x = profil_x(:, k);
    
end
























% Adding the path of data
addpath('dat');
data_dir = dir('dat');
N = size(data_dir, 1);


%load('stroke_0_0001.mat')                      % Load the data
%X= pos(:,1);                                   % Identify the x-axis data
%Y=pos(:,2);                                    % Identify the y-axis data
%fig = plot(X,Y);                               % Ploting image of x-y axis
%saveas(fig,'number.png')                       % Saveing image in formate
%x=[X,Y]';                                      % Creating an array                          % Creating logical data


% Resizing the images
l = 50;
w = 50;
data = zeros(N-2, (l*w));                       % Using l*w size 

v = [0:9];
class = (repelem(v,5))';                        % Class data

for i = 3 : N
    im = load(data_dir(i).name);
    x = im.pos(:, 1);                           % Position identify
    y = im.pos(:, 2);                           % Position identify
    %X = [x y];                                 % Same down comment
    %data(i - 2,:) = reshape(im_logical, 1, []); % Make it comment to find previous code
    
  
    
    fig = plot(x, y);
    saveas(fig,'fig.png')                       % Creating an image
    im_read = imread('fig.png');
    im_size = imresize(im_read,[l w]);          % Resize images
    %im_logical = im2bw(im_size, 0.99);         % Creating logical data
    im_2d = rgb2gray(im_size);                  % Gray scale image
    data(i - 2,:) = reshape(im_2d, 1, []);   
    disp(i);

end

%data = data / 255;
save('new_data_1.mat', 'data', 'class');               % Save the data 





% References
% Digital Imaging and Image Preprocessing course (DIIP), autumn semester, 2021.
