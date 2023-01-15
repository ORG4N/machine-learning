clc
clear

% Import dataset - read into table
filename = "cw_dataset_processed.txt";
data = readtable(filename);     % 3000 x 42 samples



% -- DATA PREPARATION START -- %

% 1. Remove duplicate samples
data = unique(data, "stable");  % 2854 x 42 samples


% 2. Remove features with same value
toRemove = {};              % Initialise array
for i=1:width(data)         % For all columns
    column = data(:,i);     % Get column at index
    
    % If 1 then all values in the column were duplicates
    if height(unique(column)) == 1
        toRemove = [toRemove, i];   % Store indexes of duplicate columns
    end
end


% Remove each column at index
for i=length(toRemove):-1:1    % Decrement because otherwise indexes will change
    index = toRemove{i};
    data(:,index) = [];        % 2854 x 34 samples
end


% 3. Normalization
w = width(data);             % Cant normalize last column (string)         
norm = data(:,1:w-1);        % So make a copy of the dataset without last col
norm = normalize(norm);      % Normalize data
data = [norm, data(:,w)];    % Add last column back to now normalized data


% 4. Shuffle the data
h = height(data);                   % Get number of rows
shuffle = data(randperm(h),:);      % Shuffle rows
data = shuffle(:,1:w-1);            % Remove last column again

% -- DATA PREPRATION COMPLETE -- %



% -- PCA -- %

dm = data{:,:};                             % table to matrix
[~, score, latent, ~, explained] = pca(dm); % PCA on matrix

figure(1);
bar(latent,'b');
title("PCA Analysis for KDD99");
xlabel("PCA Components"); ylabel("Variance");

figure(2);
plot(cumsum(explained), 'k-');
title("Cumulative Sum of Variance");
xlabel("PCA Components"); ylabel("Percentage of Variancecd do");

figure(3);
scatter(score(:,1), score(:,2),'bo');
title("Principal Component Analysis (PCA)");
xlabel("PC1"); ylabel("PC2");


