


function [lowercf,uppercf] = confidence_level(percent,ksei, g2,g3,sigma_h2, b2,b1, al, g_qub, intera, M_alpha,n,randomness,r)
	z_l = norminv(1-percent/2);
	z_u = norminv(percent/2);
	u = z_l;
	g0 = (2*u^2+1)/(6*ksei^3)*g_qub +(r-1)*(u^2+1)/(2*ksei^3)*intera;
	l_upper = floor((al/2)/(al-1));
	g_hat = zeros(l_upper,1);
	for l = 1: l_upper
		g_hat(l,1) =  g_tuda_esti(u, ksei, g2,g3, sigma_h2,b2,b1,l,randomness,r)/M_alpha^l ;
    end
	lowercf = u - g0/sqrt(n) + sum(g_hat);
	u = z_u;
	g0 = (2*u^2+1)/(6*ksei^3)*g_qub +(r-1)*(u^2+1)/(2*ksei^3)*intera;
	g_hat = zeros(l_upper,1);
	for l = 1: l_upper
		g_hat(l,1) =  g_tuda_esti(u, ksei, g2,g3,sigma_h2, b2,b1,l,randomness,r)/M_alpha^l; 
    end
	uppercf = u - g0/sqrt(n) + sum(g_hat);

	end

