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
%imshowpair(I,BW,'montage')
%x=[X,Y]';                              % Creating an array
%Bin = imbinarize(x);                   % Creating logical data




% Resizing the images
l = 25;
w = 25;
%im_resize = l * w;
data = zeros(N-2, (l*w));               % Using l*w size 

for i = 3 : N
        im = load(data_dir(i).name);
        %posit = im.pos;
        x = im.pos(:, 1);                  % Position
        y = im.pos(:, 2);                  % Position
        %figure('visible', 'off');       %to see images, comment this line
        axis auto
        fig = plot(x, y);                %plotting the image using 2 dims
        %set(gca, 'Visible', 'off');
        saveas(fig, 'fig', 'png');       %making an image object
        img = imread('fig.png');
        img = imresize(img,[l w]);      %resizing all images to one size 
        img = im2bw(img, 0.99); %imbinarize(img);            % Creating logical data
        %imshow(img);
        array_line = reshape(img, 1, []);       % Array of pixels to line
        %data(i - 2,:) = array_line;             % Saving into data table
        disp(i);
end

save('data.mat','data');