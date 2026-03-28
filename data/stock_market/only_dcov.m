addpath('subroutines');

rng(4)
w = warning ('off','all');
sectorinfo = readtable("./constituents_csv.csv", 'TextType', 'string','PreserveVariableNames',false);
uniqsector = unique(sectorinfo.Sector);
recordtstat = zeros(length(uniqsector),length(uniqsector));
recordtstatstd = zeros(length(uniqsector),length(uniqsector));
recordpval = zeros(length(uniqsector),length(uniqsector));
recordpvalstd = zeros(length(uniqsector),length(uniqsector));
itern = 1;
timecost = zeros(length(uniqsector),length(uniqsector),itern);
ksei2_all = zeros(length(uniqsector),length(uniqsector),itern);
mu_hat_all = zeros(length(uniqsector),length(uniqsector),itern);


for k = 1:(length(uniqsector))
    for m = k:length(uniqsector)
        sector1name = uniqsector(k);
        sector2name = uniqsector(m);
        sector1 = sectorinfo.Symbol(sectorinfo.Sector == sector1name,:);
        Files = dir("./orignal_data/individual_stocks");
        valueall1 = zeros(length(sector1),138);
        deletea = 0;
        j = 0;
        nameall = {Files(:).name};
        for i = 1:length(sector1)
            weneed = strcat(sector1(i),'.csv');
            if any(strcmp(nameall,weneed)) ~=1
                valueall1(i,:) = NaN;
            else
                data = readtable(strcat("./orignal_data/individual_stocks/",sector1(i)), 'TextType', 'string','PreserveVariableNames',false);
                index_start = find(data.Date=='03-01-2000');
                if isempty(index_start)
                    valueall1(i,:) = NaN;
                else
                    len_date = size(data,1);
                    if len_date < index_start +2759+19
                        deletea = deletea+1;
                        valueall1(i,:) = NaN;
                    else
                        list_date1 = index_start:20:index_start+2759;
                        list_date2 = index_start+19:20:index_start+2759+19;
                        value = log(data.AdjustedClose(list_date2)./data.AdjustedClose(list_date1));
                        valueall1(i,:) =  value';
                    end
                end
            end
        end
        valueall1 = valueall1(~isnan(valueall1(:,1)),:);
        
        
        
        
        
        sector2 = sectorinfo.Symbol(sectorinfo.Sector == sector2name,:);
        Files = dir("./orignal_data/individual_stocks");
        valueall2 = zeros(length(sector2),138);
        deletea = 0;
        j = 0;
        nameall = {Files(:).name};
        for i = 1:length(sector2)
            weneed = strcat(sector2(i),'.csv');
            if any(strcmp(nameall,weneed)) ~=1
                valueall2(i,:) = NaN;
            else
                data = readtable(strcat("./orignal_data/individual_stocks/",sector2(i)), 'TextType', 'string','PreserveVariableNames',false);
                index_start = find(data.Date=='03-01-2000');
                if isempty(index_start)
                    valueall2(i,:) = NaN;
                else
                    len_date = size(data,1);
                    if len_date < index_start +2759+19
                        deletea = deletea+1;
                        valueall2(i,:) = NaN;
                    else
                        list_date1 = index_start:20:index_start+2759;
                        list_date2 = index_start+19:20:index_start+2759+19;
                        value = log(data.AdjustedClose(list_date2)./data.AdjustedClose(list_date1));
                        valueall2(i,:) =  value';
                    end
                end
            end
        end
        
        
        valueall2 = valueall2(~isnan(valueall2(:,1)),:);
        
        data1  = valueall1';
        data2 = valueall2';
        type = {};
        type.name = "stockmarket";
        type.permu = perms(1:4);
        percent = 0.1;
        
        r = 4;
        pvalue_all = zeros(itern,1);
        tstat_all = zeros(itern,1);
        
        n = size(data1,1);
        li2 = nchoosek(1:n,r);
        for l = 1:itern
                tic;
                h_vec = motif(data1,data2,type,li2)';
                mu_hat = mean(h_vec);
                mu_hat_all(k,m,l) = mu_hat;  
                timecost(k,m,l) = toc;
        end
    end
end
save("./result/1CI_stockall5_comp","timecost","mu_hat_all")
