% Adding the path of data
addpath('training_data');
data_dir = dir('training_data');
N = size(data_dir, 1);

% Resizing the images
l = 50;
w = 50;
im_resize = l * w;
data = zeros(N - 2, im_resize);

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
        %print('img2','-dpng', '-noui')
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