
function [mu_hat,test_points,pvalue,ksei2] = CI_ustat_comp(percent,x1,x2, type,r)
%%% calculate the confidence interval
%%%% using nondegenerate method 
    


    n = size(x1,1);
    li2 = nchoosek(1:n,r);
    h_vec = motif(x1,x2,type,li2)';
    mu_hat = mean(h_vec);
    [al_value,al_number] = h_redu(li2, h_vec, n);
    al_weight = al_value./al_number;
    ksei2 = mean(( al_weight - mu_hat).^2);
    g13 = mean(( al_weight - mu_hat).^3);
    switch r
            case 2
                [g2,weight] = h_g1g1g2_2(li2,h_vec,n);
            case 3
            
                [g2,weight] = h_g1g1g2(li2,h_vec,n);
            case 4
               [g2,weight] = h_g1g1g2_4(li2,h_vec,n);
    end
    g2_up = g2./weight;
    g2_up(isnan(g2_up)) = 0;
    g112 = g112_function(al_weight,al_number,g2_up,weight,mu_hat);   
    var_all = n*nchoosek(n-1,r-1)^2/(nchoosek(n,r)^2)*ksei2;
    ksei = sqrt(ksei2); 
    test_points = mu_hat/sqrt(var_all);
    GT = cdf('Normal',test_points,0,1) + pdf('Normal',test_points,0,1).*  ...
							((2*test_points.^2+1).*g13./6./ksei^3./sqrt(n)+(r-1)*(test_points.^2+1).*g112./2./ksei^3./sqrt(n));
    
	GT = max([min([GT,1]),0]);
	[gj_l, gj_u] = confidence_level(percent,ksei, 1,1,1, 1,1, 2, g13, g112, 1,n,2,r);	
		%%%%%%%%%%%%%%%%%%
    q_h = mu_hat - sqrt(var_all)*(gj_u);
    q_l = mu_hat - sqrt(var_all)*(gj_l);
	middle_point = (q_l+ q_h)/2;
    radius =  (q_h- q_l)/2;
    pvalue = 2*min(GT, 1-GT);
end

