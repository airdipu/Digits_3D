function C = KNN1(trainclass, traindata, data, k)

    n = size(traindata, 2);
    
    % Euclidan distance
    for i = 1:size(data, 2)
        for j = 1:size(data, 2)
            d(j, i) = norm(traindata(:, j) - data(:, i));%(sum((traindata(:, j) - data(:, i)).^2))^0.5;
        end
    end
    [nn, ind] = sort(d);                      % Sorting the distances
    class = repmat(trainclass, [n, 1])';
    tr1 = class(ind);

    % Choosing k-nearest data points and the classes
    kn = nn(:, 1:k); 
    tr2 = tr1(:, 1:k);

    % Classification
    for i = 1:size(data, 2)
        for j = 1:max(trainclass)
            temp(j, i) = sum(tr2(i, :) == j); % Calculating the same class
        end
    end

[tmp, C] = max(temp);






% Reference:
% https://www.matlabcoding.com/2020/05/building-k-nearest-neighbor-algorithm.html
% https://se.mathworks.com/help/stats/fitcknn.html
% https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
% https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
% https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning
