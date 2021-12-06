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

v = [0 1 2 3 4 5 6 7 8 9];
class = (repelem(v,5))';                        % Class data

for i = 3 : N
    im = load(data_dir(i).name);
    x = im.pos(:, 1);                           % Position identify
    y = im.pos(:, 2);                           % Position identify
    fig = plot(x, y);
    saveas(fig,'number.png')                 % Creating an image
    im_read = imread('fig.png');
    im_size = imresize(im_read,[l w]);         % Resize images
    %im_logical = im2bw(im_size, 0.99);         % Creating logical data
    im_2d = rgb2gray(im_size);
    data(i - 2,:) = reshape(im_2d, 1, []);   
    disp(i);

end

%data = data / 255;
save('data.mat', 'data', 'class');               % Save the data 
%, 'class'
