function [distances] = distance(k, sz, points, data)

    % Calculate Euclidean distance between centroid node and data node
    distances = zeros(k, sz);
    
    for i=1:k
    
        centroid_node = points(i,:); % Iterate through each centroid node.
    
        for j=1:sz
            
            node = data(:,j); % Iterate through each data node within the dataset.
            node = node.';    % Transpose vector to make it row-wise
    
            d = sqrt((node - centroid_node) * (node - centroid_node)'); % Euclidean distance
            
            distances(i, j) = d;
            
        end
    end

end

