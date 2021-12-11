
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
