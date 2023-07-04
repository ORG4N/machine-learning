function [c1, c2, c3] = setcluster(sz, distances, data)

    % There are 3 clusters (k=3) and therefore we get the three Euclidean
    % distances for each data node and compare them to find the shortest.
    map = zeros(1, sz);
    
    c1 = [];
    c2 = [];
    c3 = [];
    
    for i=1:sz
    
        % The index of the shortest distance = the cluster id
        valuesToCompare = distances(:,i)';
        
        mValue = min(valuesToCompare); % Get minumum value from distance vector
        [row, col] = find(valuesToCompare == mValue); % Use Find to get col index to identify cluster
        
        map(i) = col;
    
        if col==1
            c1 = [c1 data(:,i)];
        end
    
        if col==2
            c2 = [c2 data(:,i)];
        end
    
        if col==3
            c3 = [c3 data(:,i)];
        end
    
    end
end

