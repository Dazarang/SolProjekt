% Define the output Excel file name
outputFile = 'exported_data.xlsx';

% Export cleanCustomerColumns to the first sheet
writematrix(cleanCustomerColumns, outputFile, 'Sheet', 'CustomerColumns');

% Due to Excel's limitations with large datasets, we'll split the larger matrices
% into multiple sheets if needed (Excel has a limit of 1,048,576 rows per sheet)

% Export evLoadData
numSplits = ceil(size(evLoadData, 1) / 1000000);
for i = 1:numSplits
    startRow = ((i-1) * 1000000) + 1;
    endRow = min(i * 1000000, size(evLoadData, 1));
    sheetName = sprintf('EVLoad_Part%d', i);
    writematrix(evLoadData(startRow:endRow, :), outputFile, 'Sheet', sheetName);
end

% Export generation data
numSplits = ceil(size(generation, 1) / 1000000);
for i = 1:numSplits
    startRow = ((i-1) * 1000000) + 1;
    endRow = min(i * 1000000, size(generation, 1));
    sheetName = sprintf('Generation_Part%d', i);
    writematrix(generation(startRow:endRow, :), outputFile, 'Sheet', sheetName);
end

% Export totalHouseLoad data
numSplits = ceil(size(totalHouseLoad, 1) / 1000000);
for i = 1:numSplits
    startRow = ((i-1) * 1000000) + 1;
    endRow = min(i * 1000000, size(totalHouseLoad, 1));
    sheetName = sprintf('HouseLoad_Part%d', i);
    writematrix(totalHouseLoad(startRow:endRow, :), outputFile, 'Sheet', sheetName);
end

fprintf('Data export completed to %s\n', outputFile);