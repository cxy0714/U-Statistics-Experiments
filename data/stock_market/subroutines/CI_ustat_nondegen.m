
function [mu_hat,test_points,pvalue,ksei2] = CI_ustat_nondegen(percent,x1,x2, al, type,r)
%%% calculate the confidence interval
%%%% using nondegenerate method 
    


    n = size(x1,1);
    li2 = randomsamle(n,al);
    M_alpha2 = min(floor(n^(al-1)),floor((n-1)/(r-1)));
    li_hat = [];
    for i = 1:M_alpha2
            item = zeros(n,r);
            item(:,1) = 1:n;
            for c = 1:r-1
                item(:,c+1) = item(:,1)+i*c;
            end
            item = mod(item-1,n)+1;
            li_hat = [li_hat;item];
    end
    g2_li = zeros(M_alpha2*n,r,2);
    g2_li1 = [];
    g2_li2 = [];
    for i = 1:M_alpha2
		    item = zeros(n,r);
            item(:,1) = 1:n;
		    for c = 0:r-1
			    item(:,c+1) = item(:,1)+i*c;
            end
            g2_li1 = [g2_li1;item];
		    item2 = zeros(n,r);
		    for c = 0:r-1
			    item2(:,c+1) = item(:,end)+i*(c-1);
		    end
		    g2_li2 = [g2_li2;item2];
    end
    g2_li(:,:,1) = g2_li1;
    g2_li(:,:,2) = g2_li2;
    g2_li = mod(g2_li-1,n)+1;

    h_vec = motif(x1,x2,type,li2)';
    h_vec2 = motif(x1,x2,type,li_hat)';
    mu_hat_esti = mean(h_vec2);
    mu_2 = mu_hat_esti^2;
    sigma_h2 = mean(h_vec2.^2)-mu_2;
    mu_hat = mean(h_vec);
    [al_value,al_number] = h_redu(li2, h_vec, n);
    al_weight = al_value./al_number;
    ksei2 = mean(( al_weight - mu_hat).^2);
	ksei = sqrt(ksei2);
    g13 = mean(( al_weight - mu_hat).^3);
    var_all = sum(al_number.^2)/(floor(n^al)^2)*ksei2;
    g112 = 1/size(li2,1)*sum((h_vec'-mu_hat-(al_weight(li2(:,1))-mu_hat)-(al_weight(li2(:,r))-mu_hat)).*(al_weight(li2(:,1))-mu_hat).*(al_weight(li2(:,r))-mu_hat));    

	xx6 = motif(x1,x2,type,g2_li(:,:,1)); xx6 = xx6 - mean(xx6)-(al_weight(g2_li(:,(r-1),1))-mu_hat)-(al_weight(g2_li(:,r,1))-mu_hat);
	xx7 = motif(x1,x2,type,g2_li(:,:,2)); xx7 = xx7 - mean(xx7)-(al_weight(g2_li(:,1,2))-mu_hat)-(al_weight(g2_li(:,2,2))-mu_hat);
	g2 = mean(xx6.*xx7) ;
	M_alpha_new = n^(al-1)*(1+1/(r*n^(al-1)))/(1+n^(al-2)*g2*r*(r-1)/(sigma_h2-r*ksei^2));

    test_points = mu_hat/sqrt(var_all);
    GT = cdf('Normal',test_points,0,1) + pdf('Normal',test_points,0,1).*  ...
							((2*test_points.^2+1).*g13./6./ksei^3./sqrt(n)+(r-1)*(test_points.^2+1).*g112./2./ksei^3./sqrt(n)-(sigma_h2-r*ksei^2)/(r^2*ksei^2).*(test_points)./2/M_alpha_new);
    
	GT = max([min([GT,1]),0]);
	[gj_l, gj_u] = confidence_level(percent,ksei, 1,1,sigma_h2, 1,1, al, g13, g112, M_alpha_new,n,1,r);	
		%%%%%%%%%%%%%%%%%%
    q_h = mu_hat - sqrt(var_all)*(gj_u);
    q_l = mu_hat - sqrt(var_all)*(gj_l);
	middle_point = (q_l+ q_h)/2;
    radius =  (q_h- q_l)/2;
    pvalue = 2*min(GT, 1-GT);
end

