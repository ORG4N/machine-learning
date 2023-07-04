% Mean and standard deviation for each cluster
m1 = [-4;-1];   std1 = 0.75;
m2 = [3;4];     std2 = 2.0;
m3 = [-2;10];   std3 = 0.3;

N = 1000;   % Sample size (for each cluster)
k = 3;      % Centroids (set to 3 because there are 3 clusters) 
ITERATIONS = 15;

d1 = std1 * randn(2, N) + repmat(m1, 1, N); % 2D Gausssian cluster 1 
d2 = std2 * randn(2, N) + repmat(m2, 1, N); % 2D Gausssian cluster 2
d3 = std3 * randn(2, N) + repmat(m3, 1, N); % 2D Gausssian cluster 3

data = [d1 d2 d3]; % Merge the three cluster classes into one dataset
sz = size(data, 2); % Find how many coord pairs are within dataset

% Plot the data with colours (unmerged)
figure
hold on
plot(d1(1,:), d1(2,:), 'r.');
plot(d2(1,:), d2(2,:), 'g.');
plot(d3(1,:), d3(2,:), 'b.');
xlabel('x-dimension');
ylabel('y-dimension');
title('original clusters');

points = zeros(k,2);  % Initial K centroid coordinates array
previously_selected = zeros(1,k); % Store chosen K coordinate indexes within Dataset

% Step 1: ( Get initial centroid nodes)
for i=1:k
    % Get a random value from dataset (Forgy method)

    while true

        init_k = randi([0, sz]); % Randomly choose X,Y coord
        
        % Don't want multiple centroids with same starting value
        if ismember(init_k, previously_selected) == 0

            points(i,1) = data(1, init_k); % Store X 
            points(i,2) = data(2, init_k); % Store Y
            previously_selected(i) = init_k;
            break
        end
        
        fprintf("%i already selected as a centroid\n", init_k);

    end
end

points_history = zeros(k,2);  % Centroid coordinates array history

for i=1:ITERATIONS
    % Step 2a: (Call distance function to calculate Euclidian distances.)
    distances = distance(k, sz, points, data);
    
    % Step 2b: (Find closest centroid nodes to each data node, to identify clusters)
    [c1, c2, c3] = setcluster(sz, distances, data);

    if i==1 | i==round(ITERATIONS/2)
        % Draw the graph to show coloured clusters and centroids
        figure
        hold on
        plot(c1(1,:), c1(2,:), 'c.');
        plot(c2(1,:), c2(2,:), 'g.');
        plot(c3(1,:), c3(2,:), 'y.');
        plot(points(:,1),points(:,2), '+', 'MarkerSize', 10, 'LineWidth',2);
        xlabel('x-dimension');
        ylabel('y-dimension');
        title(['K-means Clustering (Iteration =', num2str(i), ')']);
        legend('class1', 'class2', 'class3', 'Nodes');
    end 
    
    
    % Store points before changing them
    points_history = points;

    % Step 3: (recalculate centroids within clusters)
    points(1,:) = [mean(c1(1,:)), mean(c1(2,:))];
    points(2,:) = [mean(c2(1,:)), mean(c2(2,:))];
    points(3,:) = [mean(c3(1,:)), mean(c3(2,:))];

    % If this iteration resulted in the same centroids as the previous then
    % stop iterating.
    if points_history == points
        fprintf("Iteration %i - convergence reached, no change recorded.\n", i);
        break
    end
end


% Final optimized graph 
% Will always output to show converged result
figure
hold on
plot(c1(1,:), c1(2,:), 'c.');
plot(c2(1,:), c2(2,:), 'g.');
plot(c3(1,:), c3(2,:), 'y.');
plot(points(:,1),points(:,2), '+', 'MarkerSize', 10, 'LineWidth',2);
xlabel('x-dimension');
ylabel('y-dimension');
title(['K-means Clustering Final Iteration - ', num2str(i) ]);
legend('class1', 'class2', 'class3', 'Nodes');