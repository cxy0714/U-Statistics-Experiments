% S&P 500 Log Monthly Returns Processing Script
% This script processes individual stock data from S&P 500 constituents
% and creates a 3D tensor of log monthly returns organized by sector

% Add subroutines to path and set random seed for reproducibility
addpath('subroutines');
rng(4)
w = warning ('off','all');

% Read sector information from CSV file
% This file contains stock symbols and their corresponding sectors
sectorinfo = readtable("./constituents_csv.csv", 'TextType', 'string','PreserveVariableNames',false);
uniqsector = unique(sectorinfo.Sector);

% Initialize tensor dimensions
num_sectors = length(uniqsector);
max_stocks_per_sector = 44;  % Maximum number of stocks per sector
num_time_points = 138;       % Number of monthly time points

% Initialize 3D tensor: [sectors, stocks_per_sector, time_points]
log_monthly_returns = NaN(num_sectors, max_stocks_per_sector, num_time_points);

% Initialize storage for stock names and counts per sector
sector_stock_names = cell(num_sectors, max_stocks_per_sector);
sector_stock_counts = zeros(num_sectors, 1);

% Get list of all available stock data files
Files = dir("./orignal_data/individual_stocks");
nameall = {Files(:).name};

fprintf('Processing sector stock data...\n');

% Process each sector separately
for k = 1:num_sectors
    sector_name = uniqsector(k);
    fprintf('Processing sector %d: %s\n', k, sector_name);
    
    % Get all stocks belonging to current sector
    sector_stocks = sectorinfo.Symbol(sectorinfo.Sector == sector_name, :);
    
    % Temporary storage for current sector data
    temp_data = [];
    temp_names = {};
    valid_count = 0;
    
    % Process each stock in the current sector
    for i = 1:length(sector_stocks)
        stock_symbol = sector_stocks(i);
        weneed = strcat(stock_symbol, '.csv');
        
        % Check if stock data file exists
        if any(strcmp(nameall, weneed)) ~= 1
            continue;
        end
        
        try
            % Read stock price data
            data = readtable(strcat("./orignal_data/individual_stocks/", stock_symbol), ...
                           'TextType', 'string', 'PreserveVariableNames', false);
            
            % Find starting date (March 1, 2000)
            index_start = find(data.Date == '03-01-2000');
            
            if isempty(index_start)
                continue;
            end
            
            % Check if we have enough data points
            len_date = size(data, 1);
            if len_date < index_start + 2759 + 19
                continue;
            end
            
            % Calculate monthly sampling indices (every 20 trading days ≈ 1 month)
            list_date1 = index_start:20:index_start+2759;        % Start dates
            list_date2 = index_start+19:20:index_start+2759+19;  % End dates
            
            % Calculate log returns for monthly periods
            log_returns = log(data.AdjustedClose(list_date2) ./ data.AdjustedClose(list_date1));
            
            % Store valid stock data
            valid_count = valid_count + 1;
            temp_data = [temp_data; log_returns'];
            temp_names{end+1} = stock_symbol;
            
        catch ME
            fprintf('Error processing stock %s: %s\n', stock_symbol, ME.message);
            continue;
        end
    end
    
    % Store sector data in main tensor
    if valid_count > 0
        log_monthly_returns(k, 1:valid_count, :) = temp_data;
        sector_stock_names(k, 1:valid_count) = temp_names;
        sector_stock_counts(k) = valid_count;
    end
    
    fprintf('Sector %s completed, valid stocks: %d\n', sector_name, valid_count);
end

% Display processing summary
fprintf('\n=== Data processing completed ===\n');
fprintf('Total sectors processed: %d\n', num_sectors);
fprintf('Tensor dimensions: [%d, %d, %d]\n', size(log_monthly_returns));
fprintf('\nNumber of stocks per sector:\n');
for k = 1:num_sectors
    fprintf('%s: %d stocks\n', uniqsector(k), sector_stock_counts(k));
end

% Create sector index mapping for easy lookup
sector_index_map = containers.Map(cellstr(uniqsector), num2cell(1:num_sectors));

% Generate time labels for each time point
start_date = datetime('2000-03-01');
time_points = start_date + calmonths(0:(num_time_points-1));
time_labels = cellstr(datestr(time_points, 'yyyy-mm'));

% Create comprehensive index mapping table
fprintf('\n=== Creating index mapping table ===\n');
index_mapping = [];
for sector_idx = 1:num_sectors
    for stock_idx = 1:sector_stock_counts(sector_idx)
        mapping_entry = struct();
        mapping_entry.sector_idx = sector_idx;
        mapping_entry.stock_idx = stock_idx;
        mapping_entry.sector_name = char(uniqsector(sector_idx));
        mapping_entry.stock_symbol = char(sector_stock_names{sector_idx, stock_idx});
        index_mapping = [index_mapping; mapping_entry];
    end
end

% Create output directory if it doesn't exist
if ~exist('matlab_data', 'dir')
    mkdir('matlab_data');
end

% Save main MATLAB data file with all variables
save_filename = 'matlab_data/sp500_log_monthly_returns_tensor.mat';
save(save_filename, 'log_monthly_returns', 'sector_stock_names', 'sector_stock_counts', ...
     'uniqsector', 'sector_index_map', 'num_sectors', 'max_stocks_per_sector', 'num_time_points', ...
     'time_labels', 'time_points', 'index_mapping');

fprintf('\nData saved to: %s\n', save_filename);

% Display sample index mapping information
fprintf('\n=== Index mapping table example ===\n');
fprintf('First 10 valid stock index information:\n');
fprintf('%-8s %-8s %-25s %-10s\n', 'SectorID', 'StockID', 'SectorName', 'StockSymbol');
fprintf('%-8s %-8s %-25s %-10s\n', '--------', '-------', '-----------', '-----------');
for i = 1:min(10, length(index_mapping))
    fprintf('%-8d %-8d %-25s %-10s\n', ...
        index_mapping(i).sector_idx, index_mapping(i).stock_idx, ...
        index_mapping(i).sector_name, index_mapping(i).stock_symbol);
end

% ========== UTILITY FUNCTIONS ==========

% Function to get detailed information about a specific tensor position
function info = get_tensor_info(tensor_idx1, tensor_idx2, tensor_idx3, sector_stock_names, uniqsector, time_labels)
    % Get information for a specific tensor position [sector, stock, time]
    info = struct();
    info.sector_idx = tensor_idx1;
    info.stock_idx = tensor_idx2;
    info.time_idx = tensor_idx3;
    
    % Get sector name
    if tensor_idx1 <= length(uniqsector)
        info.sector_name = char(uniqsector(tensor_idx1));
    else
        info.sector_name = 'Invalid';
    end
    
    % Get stock symbol
    if tensor_idx1 <= size(sector_stock_names, 1) && tensor_idx2 <= size(sector_stock_names, 2)
        stock_name = sector_stock_names{tensor_idx1, tensor_idx2};
        if ~isempty(stock_name)
            info.stock_symbol = char(stock_name);
        else
            info.stock_symbol = 'Empty';
        end
    else
        info.stock_symbol = 'Invalid';
    end
    
    % Get time label
    if tensor_idx3 <= length(time_labels)
        info.time_label = time_labels{tensor_idx3};
    else
        info.time_label = 'Invalid';
    end
    
    % Display information
    fprintf('Tensor position (%d,%d,%d) corresponds to:\n', tensor_idx1, tensor_idx2, tensor_idx3);
    fprintf('  Sector: %s (index %d)\n', info.sector_name, info.sector_idx);
    fprintf('  Stock: %s (index %d)\n', info.stock_symbol, info.stock_idx);
    fprintf('  Time: %s (index %d)\n', info.time_label, info.time_idx);
end

% Function to find tensor indices for a specific stock symbol
function indices = find_stock_indices(stock_symbol, sector_stock_names)
    % Find the tensor position [sector_idx, stock_idx] for a given stock symbol
    indices = [];
    for i = 1:size(sector_stock_names, 1)
        for j = 1:size(sector_stock_names, 2)
            if ~isempty(sector_stock_names{i, j}) && strcmp(sector_stock_names{i, j}, stock_symbol)
                indices = [i, j];
                return;
            end
        end
    end
    if isempty(indices)
        fprintf('Stock not found: %s\n', stock_symbol);
    else
        fprintf('Stock %s is at tensor position: (%d, %d, :)\n', stock_symbol, indices(1), indices(2));
    end
end

% Function to extract all data for a specific sector
function [data, names] = get_sector_data(log_monthly_returns, sector_stock_names, sector_stock_counts, sector_idx)
    % Extract all stock data for a specific sector
    if sector_idx > size(log_monthly_returns, 1)
        error('Sector index out of range');
    end
    
    valid_stocks = sector_stock_counts(sector_idx);
    data = log_monthly_returns(sector_idx, 1:valid_stocks, :);
    names = sector_stock_names(sector_idx, 1:valid_stocks);
    
    fprintf('Sector %d contains %d stocks\n', sector_idx, valid_stocks);
end

% ========== USAGE EXAMPLES ==========

fprintf('\n=== Usage examples ===\n');

% Example 1: Query tensor position information
fprintf('1. Query information for tensor position (1,2,3):\n');
if sector_stock_counts(1) >= 2
    info = get_tensor_info(1, 2, 3, sector_stock_names, uniqsector, time_labels);
    log_return_value = log_monthly_returns(1, 2, 3);
    fprintf('  Corresponding log return value: %.6f\n', log_return_value);
end

% Example 2: Find stock position in tensor
fprintf('\n2. Find tensor position for specific stock:\n');
if sector_stock_counts(1) > 0
    example_stock = sector_stock_names{1, 1};
    fprintf('Looking for stock: %s\n', example_stock);
    indices = find_stock_indices(example_stock, sector_stock_names);
end

% Example 3: Display time labels
fprintf('\n3. Time labels example (first 10):\n');
for i = 1:min(10, length(time_labels))
    fprintf('  Time index %d: %s\n', i, time_labels{i});
end

% Example 4: Display stocks in first sector
fprintf('\n4. All stocks in first sector:\n');
if sector_stock_counts(1) > 0
    fprintf('Sector: %s\n', uniqsector(1));
    for i = 1:sector_stock_counts(1)
        fprintf('  Stock %d: %s\n', i, sector_stock_names{1, i});
    end
end

% ========== EXPORT DATA FOR PYTHON ==========

% Save tensor data in binary format for Python
tensor_filename = 'matlab_data/log_return_tensor.dat';
fid = fopen(tensor_filename, 'wb');
fwrite(fid, size(log_monthly_returns), 'int32');  % Write dimensions first
fwrite(fid, log_monthly_returns(:), 'double');    % Write tensor data
fclose(fid);
fprintf('\nTensor data saved to: %s\n', tensor_filename);

% Create and save sector information table for Python
sector_info_table = table();
sector_info_table.Sector = uniqsector;
sector_info_table.SectorIndex = (1:length(uniqsector))';
sector_info_table.StockCount = sector_stock_counts;
sector_csv_filename = 'matlab_data/python_sector_info.csv';
writetable(sector_info_table, sector_csv_filename);
fprintf('Sector info saved to: %s\n', sector_csv_filename);

% Create and save stock information table for Python
stock_info_table = table();
stock_info_rows = [];
for i = 1:num_sectors
    for j = 1:sector_stock_counts(i)
        row = struct();
        row.SectorIndex = i;
        row.StockIndex = j;
        row.Sector = uniqsector(i);
        row.StockSymbol = sector_stock_names{i, j};
        stock_info_rows = [stock_info_rows; row];
    end
end

% Convert struct array to table and save
if ~isempty(stock_info_rows)
    stock_info_table.SectorIndex = [stock_info_rows.SectorIndex]';
    stock_info_table.StockIndex = [stock_info_rows.StockIndex]';
    stock_info_table.Sector = {stock_info_rows.Sector}';
    stock_info_table.StockSymbol = {stock_info_rows.StockSymbol}';
    stock_csv_filename = 'matlab_data/python_stock_info.csv';
    writetable(stock_info_table, stock_csv_filename);
    fprintf('Stock info saved to: %s\n', stock_csv_filename);
end

fprintf('\n=== All files saved to matlab_data directory ===\n');