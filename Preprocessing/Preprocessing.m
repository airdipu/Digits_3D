% Adding the path of data
addpath('dat');
data_dir = dir('dat');
N = size(data_dir, 1);


%load('stroke_0_0001.mat')              % Load the data
%X= pos(:,1);                           % Identify the x-axis data
%Y=pos(:,2);                            % Identify the y-axis data
%fig = plot(X,Y);                       % Plot the image of x-y axis
%saveas(fig,'number.png')               % Save the image in a formate
%figure
%x=[X,Y]';                              % Creating an array
%bin = imbinarize(x);                   % Creating logical data
%imshowpair(fig, bin,'montage')




% Resizing the images
l = 25;
w = 25;
data = zeros(N-2, (l*w));               % Using l*w size 

for i = 3 : N
        im = load(data_dir(i).name);
        x = im.pos(:, 1);                  % Position
        y = im.pos(:, 2);                  % Position
        fig = plot(x, y);                %plotting the image using 2 dims
        saveas(fig, 'fig', 'png');       %making an image object
        im_read = imread('fig.png');
        im_size = imresize(im_read,[l w]);      %resizing all images to one size 
        im_logical = im2bw(im_size, 0.99); %imbinarize(img);            % Creating logical data
        array_line = reshape(im_logical, 1, []);       % Array of pixels to line
        %data(i - 2,:) = array_line;             % Saving into data table
        disp(i);
end

save('data.mat','data');