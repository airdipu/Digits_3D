% Adding the path of data
addpath('dat');
data_dir = dir('dat');
N = size(data_dir, 1);


%load('stroke_0_0001.mat')
%X= pos(:,1);
%Y=pos(:,2);
%fig = plot(X,Y);
%saveas(fig,'number.jpg')
%x=[X,Y]';
%Bin = imbinarize(x);




% Resizing the images
l = 20;
w = 20;
im_resize = l * w;
data = zeros(N-2, im_resize);

for i = 3 : N
        im = load(data_dir(i).name);
        posit = im.pos;
        x = posit(:, 1);                  % Position
        y = posit(:, 2);                  % Position
        figure('visible', 'off');       %to see images, comment this line
        axis auto
        f1 = plot(x, y);                %plotting the image using 2 dims
        set(gca, 'Visible', 'off');
        saveas(f1, 'img', 'png');       %making an image object
        img = imread('img.png');
        img = imresize(img,[l w]);      %resizing all images to one size 
        %img = rgb2gray(img);           %1)grayscale preprocessing
        img = im2bw(img, 0.99);         %2)binarization
        imshow(img);
        trans_data = reshape(img, 1, []);       %transform array of pixels into line
        data(i - 2,:) = trans_data;             %saving each image into data table
        disp(i);
end
% data = data / 255;                     %scaling for grayscale image
save('data.mat','data');