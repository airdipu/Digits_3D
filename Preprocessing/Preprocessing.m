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

        % Rotation around x (Using DIIP)
        rotate_around = [cos(theta) 0 sin(theta); 0 1 0; - sin(theta) 0 ...
            cos(theta)];

        data_rotation = inputdata(1:stroke_size(j), :, j)*rotate_around;
        
        n_rotation_x_data(1:size(data_rotation, 1), :, ...
            (i-1)*size(data_in, 3) + j) = bsxfun(@minus, data_rotation, ...
            min(data_rotation, [], 1))./repmat((max(data_rotation, [] , ...
            1) - min(data_rotation, [], 1)), size(data_rotation, 1), 1);
        
        % Each step of j one new matrix is adapted from the zeros matrix in the
        % third dimension of the data_rotation matrix
    end
end

n_rotation_x_data = repmat(data_class, size(rotation_angle, 2), 1);

save ('n_rotation_x_data.mat','n_rotation_x_data')
save ('stroke_size_x_rotation.mat','stroke_size_x_rotation')
save ('n_rotation_x_data.mat','n_rotation_x_data')





















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
% https://se.mathworks.com/matlabcentral/newsreader/view_thread/308859.
% Digital Imaging and Image Preprocessing course (DIIP), autumn semester, 2021.
