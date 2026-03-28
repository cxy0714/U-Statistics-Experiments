addpath('subroutines');

alpha_all = [1.5,2,2.5];
for alindex = 1:length(alpha_all)
    al = alpha_all(alindex);
    load(sprintf("./result/CI_stockall5%d",al*100))
    w = warning ('off','all');
    sectorinfo = readtable("./data/constituents_csv.csv", 'TextType', 'string','PreserveVariableNames',false);
    uniqsector = unique(sectorinfo.Sector);
    A = ones(size(recordtstat))*(-100);
    B = tril(A,-1);
    recordtstat = recordtstat +B;
    sectorred = ["CmS","CD","CnS","E","F","HC","I","IT","M","RE","U"];
    text_all = strings(1,length(uniqsector));
    for i = 1:length(uniqsector)
        text_all(i) = strcat(sectorred(i),':'," ",uniqsector(i));
    end
    textx = [ones(1,6)*1,ones(1,5)*4];
    texty = [7.5:0.5:10,9:0.5:11];
    
    titlesize =18;
    titlemulti = 2;
    fig = figure("visible","on");
    mm = 101;
    half_mm = floor(mm/2);
    cmp = zeros(mm,3);
    portions = (0:(half_mm-1))/half_mm;  portions = portions(:);
    cmp = [...
        portions * [1 1 1] + (1-portions) * [0 0 1]; ...
        [1 1 1]; ...
        (1-portions) * [1 1 1] + portions * [1 0 0]; ...
        ];
    
    cmp (1, :) = [0.941 0.941 0.941];
    cp = colormap( cmp );    
    [R, C] = ndgrid(1:size(recordtstat,1), 1:size(recordtstat,2));
    R = R(:); C = C(:);
    imagesc(recordtstat);
    vals = round(recordtstat(:),2);
    valstd = round(recordtstatstd(:),2);
    mask12 = vals <= 1.96;
    mask11 = vals >-100;
    mask1 = mask11.*mask12;
    mask1 = logical(mask1);
    mask2 = vals > 1.96;
    for i = 1:size(recordtstat,1)
        for j = 1:i
            text(C((i-1)*size(recordtstat,1)+j), R((i-1)*size(recordtstat,1)+j), {string(vals((i-1)*size(recordtstat,1)+j)),strcat("(",string(valstd((i-1)*size(recordtstat,1)+j)),")")}, ...
                'color', 'k', 'HorizontalAlignment','center','FontSize', 15)
        end
    end
    text(textx,texty,text_all,'FontSize', 18)
    colorbar;
    title({'Pairwise dependency, test statistic', sprintf('Our method, $\\alpha = %1.1f$',string(al))},'interpreter','latex');
    xt = 1:1:length(uniqsector);
    xtlbl = sectorred;                     % New 'XTickLabel' Vector
    set(gca, 'XTick',xt, 'XTickLabel',xtlbl, 'XAxisLocation', 'top','fontsize',18)   % Label Ticks
    set(gca, 'YTick',xt, 'YTickLabel',xtlbl,'fontsize',18)   % Label Ticks
    ax = gca;
    ax.TitleFontSizeMultiplier = titlemulti;
    caxis([0,5.5]);
    pbaspect([1 1 1]);
    set(gca,'units','centimeters')
    set(gcf,'units','centimeters')
    pos = get(gca,'Position');
    ti = get(gca,'TightInset');
    set(gca, 'Position',[1.65 -1 22 25]);
    set(gcf, 'Units', 'Inches', 'Position', [0 0 10.1 10.5], 'PaperUnits', 'Inches', 'PaperSize', [10, 10])
    saveas(fig,strcat('./plot/heatmat_stock_new2',string(al*100),'.png'))
    sum(sum(mean(timecost,3)))
end