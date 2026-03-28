addpath('subroutines');
rng(4)
w = warning ('off','all');

% 读取板块信息
sectorinfo = readtable("./data/constituents_csv.csv", 'TextType', 'string','PreserveVariableNames',false);
uniqsector = unique(sectorinfo.Sector);


% 初始化3维张量和名称存储
% 维度: [板块数, 最大股票数, 时间点数]
num_sectors = length(uniqsector);
max_stocks_per_sector = 44;  % 最大股票数量
num_time_points = 138;

% 3维张量存储log月收益率
log_monthly_returns = NaN(num_sectors, max_stocks_per_sector, num_time_points);

% 存储名称信息
sector_stock_names = cell(num_sectors, max_stocks_per_sector);  % 股票名称
sector_stock_counts = zeros(num_sectors, 1);  % 每个板块的有效股票数量

% 获取文件列表
Files = dir("./data/individual_stocks");
nameall = {Files(:).name};

fprintf('开始处理各板块股票数据...\n');

% 处理每个板块
for k = 1:num_sectors
    sector_name = uniqsector(k);
    fprintf('处理板块 %d: %s\n', k, sector_name);
    
    % 获取当前板块的所有股票
    sector_stocks = sectorinfo.Symbol(sectorinfo.Sector == sector_name, :);
    
    % 临时存储当前板块数据
    temp_data = [];
    temp_names = {};
    valid_count = 0;
    
    % 处理每只股票
    for i = 1:length(sector_stocks)
        stock_symbol = sector_stocks(i);
        weneed = strcat(stock_symbol, '.csv');
        
        if any(strcmp(nameall, weneed)) ~= 1
            continue;  % 跳过不存在的文件
        end
        
        try
            % 读取股票数据
            data = readtable(strcat("./data/individual_stocks/", stock_symbol), ...
                           'TextType', 'string', 'PreserveVariableNames', false);
            
            % 查找起始日期
            index_start = find(data.Date == '03-01-2000');
            
            if isempty(index_start)
                continue;  % 跳过没有起始日期的股票
            end
            
            % 检查数据长度
            len_date = size(data, 1);
            if len_date < index_start + 2759 + 19
                continue;  % 跳过数据不足的股票
            end
            
            % 计算月收益率的日期索引
            list_date1 = index_start:20:index_start+2759;
            list_date2 = index_start+19:20:index_start+2759+19;
            
            % 计算log月收益率
            log_returns = log(data.AdjustedClose(list_date2) ./ data.AdjustedClose(list_date1));
            
            % 存储有效数据
            valid_count = valid_count + 1;
            temp_data = [temp_data; log_returns'];
            temp_names{end+1} = stock_symbol;
            
        catch ME
            fprintf('处理股票 %s 时出错: %s\n', stock_symbol, ME.message);
            continue;
        end
    end
    
    % 将当前板块数据存入3维张量
    if valid_count > 0
        log_monthly_returns(k, 1:valid_count, :) = temp_data;
        sector_stock_names(k, 1:valid_count) = temp_names;
        sector_stock_counts(k) = valid_count;
    end
    
    fprintf('板块 %s 完成，有效股票数: %d\n', sector_name, valid_count);
end

% 显示结果统计
fprintf('\n=== 数据处理完成 ===\n');
fprintf('总共处理 %d 个板块\n', num_sectors);
fprintf('张量维度: [%d, %d, %d]\n', size(log_monthly_returns));
fprintf('\n各板块股票数量:\n');
for k = 1:num_sectors
    fprintf('%s: %d 只股票\n', uniqsector(k), sector_stock_counts(k));
end

% 创建板块索引映射
sector_index_map = containers.Map(cellstr(uniqsector), num2cell(1:num_sectors));

% 创建时间索引 (假设从2000年3月开始，每月一个数据点)
start_date = datetime('2000-03-01');
time_points = start_date + calmonths(0:(num_time_points-1));
time_labels = cellstr(datestr(time_points, 'yyyy-mm'));

% 创建完整的索引映射表
fprintf('\n=== 创建索引映射表 ===\n');
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

% 保存数据
save_filename = 'sp500_log_monthly_returns_tensor.mat';
save(save_filename, 'log_monthly_returns', 'sector_stock_names', 'sector_stock_counts', ...
     'uniqsector', 'sector_index_map', 'num_sectors', 'max_stocks_per_sector', 'num_time_points', ...
     'time_labels', 'time_points', 'index_mapping');

fprintf('\n数据已保存到: %s\n', save_filename);

% 显示索引映射表示例
fprintf('\n=== 索引映射表示例 ===\n');
fprintf('前10个有效股票的索引信息:\n');
fprintf('%-8s %-8s %-25s %-10s\n', '板块ID', '股票ID', '板块名称', '股票代码');
fprintf('%-8s %-8s %-25s %-10s\n', '------', '------', '----------', '--------');
for i = 1:min(10, length(index_mapping))
    fprintf('%-8d %-8d %-25s %-10s\n', ...
        index_mapping(i).sector_idx, index_mapping(i).stock_idx, ...
        index_mapping(i).sector_name, index_mapping(i).stock_symbol);
end

% 查询函数：根据张量索引获取完整信息
function info = get_tensor_info(tensor_idx1, tensor_idx2, tensor_idx3, sector_stock_names, uniqsector, time_labels)
    info = struct();
    info.sector_idx = tensor_idx1;
    info.stock_idx = tensor_idx2;
    info.time_idx = tensor_idx3;
    
    if tensor_idx1 <= length(uniqsector)
        info.sector_name = char(uniqsector(tensor_idx1));
    else
        info.sector_name = 'Invalid';
    end
    
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
    
    if tensor_idx3 <= length(time_labels)
        info.time_label = time_labels{tensor_idx3};
    else
        info.time_label = 'Invalid';
    end
    
    fprintf('张量位置 (%d,%d,%d) 对应:\n', tensor_idx1, tensor_idx2, tensor_idx3);
    fprintf('  板块: %s (索引 %d)\n', info.sector_name, info.sector_idx);
    fprintf('  股票: %s (索引 %d)\n', info.stock_symbol, info.stock_idx);
    fprintf('  时间: %s (索引 %d)\n', info.time_label, info.time_idx);
end

% 反向查询函数：根据股票代码查找张量索引
function indices = find_stock_indices(stock_symbol, sector_stock_names)
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
        fprintf('未找到股票: %s\n', stock_symbol);
    else
        fprintf('股票 %s 位于张量位置: (%d, %d, :)\n', stock_symbol, indices(1), indices(2));
    end
end

% 辅助函数：获取特定板块的所有股票数据
function [data, names] = get_sector_data(log_monthly_returns, sector_stock_names, sector_stock_counts, sector_idx)
    if sector_idx > size(log_monthly_returns, 1)
        error('板块索引超出范围');
    end
    
    valid_stocks = sector_stock_counts(sector_idx);
    data = log_monthly_returns(sector_idx, 1:valid_stocks, :);
    names = sector_stock_names(sector_idx, 1:valid_stocks);
    
    fprintf('板块 %d 包含 %d 只股票\n', sector_idx, valid_stocks);
end

% 示例用法
fprintf('\n=== 使用示例 ===\n');

% 示例1：查询张量位置(1,2,3)对应的信息
fprintf('1. 查询张量位置(1,2,3)的信息:\n');
if sector_stock_counts(1) >= 2
    info = get_tensor_info(1, 2, 3, sector_stock_names, uniqsector, time_labels);
    log_return_value = log_monthly_returns(1, 2, 3);
    fprintf('  对应的log收益率值: %.6f\n', log_return_value);
end

% 示例2：查询特定股票的张量位置
fprintf('\n2. 查询特定股票的张量位置:\n');
if sector_stock_counts(1) > 0
    example_stock = sector_stock_names{1, 1};
    fprintf('查找股票: %s\n', example_stock);
    indices = find_stock_indices(example_stock, sector_stock_names);
end

% 示例3：显示所有时间标签
fprintf('\n3. 时间标签示例 (前10个):\n');
for i = 1:min(10, length(time_labels))
    fprintf('  时间索引 %d: %s\n', i, time_labels{i});
end

% 示例4：获取特定板块所有股票名称
fprintf('\n4. 第1个板块的所有股票:\n');
if sector_stock_counts(1) > 0
    fprintf('板块: %s\n', uniqsector(1));
    for i = 1:sector_stock_counts(1)
        fprintf('  股票 %d: %s\n', i, sector_stock_names{1, i});
    end
end