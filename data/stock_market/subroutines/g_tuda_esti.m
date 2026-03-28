


function [value] = g_tuda_esti(u, ksei, g2,g3,sigma_h2, b2,b1,l,randomness,r)
    switch randomness
    case 0
        switch l
    	    case 1
		        value = (sigma_h2-r*ksei^2)/((b2-b1)*r^2*ksei^2) * u/(2);
%                 value = (3*g2+g3)/((b2-b1)*r^2*ksei^2) * u/(2);
            case 2
	            value1 = (sigma_h2-r*ksei^2)/((b2-b1)*r^2*ksei^2) * u/(2);
		        dify = 1/sqrt(2*pi)*exp(-1/2*u ^2)* (-u);	
                dify1 = (sigma_h2-r*ksei^2)/((b2-b1)*r^2*ksei^2) /(2);
                value2 = ((sigma_h2-r*ksei^2)/((b2-b1)*r^2*ksei^2))^2 * (u^3-3*u)/(factorial(2*2));
                value = (value1*value1*dify/2-value2-value1*dify1)/(1+normpdf(u)*value1);

            case 3
                psi1 = g_tuda_esti(u, ksei, g2,g3,sigma_h2, b2,b1,1,randomness,r);
                psi2 = g_tuda_esti(u, ksei, g2,g3,sigma_h2, b2,b1,2,randomness,r);
		        syms z 
		        f = 1/sqrt(2*pi)*exp(-1/2*z ^2);
		        dify = diff(f);	
                dify2 = diff(dify);
                gamma1 = (sigma_h2-r*ksei^2)/((b2-b1)*r^2*ksei^2) * z /(2);
                difg1 = diff(gamma1);
                gamma2 = ((sigma_h2-r*ksei^2)/((b2-b1)*r^2*ksei^2))^2 * (z ^3-3*z)/(factorial(2*2));
                difg2 = diff(gamma2);
                gamma3 = ((sigma_h2-r*ksei^2)/((b2-b1)*r^2*ksei^2))^3 * (u ^5-10*u^3+15*u)/(factorial(2*3));
                syms vs
                eqn = vs == psi1*psi2*vpa(subs(dify,z,u))/2+psi2*psi1*vpa(subs(dify,z,u))/2+psi1^3*vpa(subs(dify2,z,u))/factorial(3)...
                            -(gamma3+psi2*vpa(subs(difg1,z,u))+psi1*vpa(subs(difg2,z,u)))-(vs*normpdf(u))*(vpa(subs(gamma2,z,u))+psi2*vpa(subs(difg1,z,u)))...
                                -(vs*normpdf(u)+psi1*psi2*vpa(subs(dify,z,u))/2+psi1*psi2*vpa(subs(dify,z,u))/2)*vpa(subs(gamma1,z,u));
                value = solve(eqn,vs);

        end
    case 1
        switch l
            case 1
		        value = (sigma_h2-r*ksei^2)/(r^2*ksei^2) * u/(2);
%                 value = (3*g2+g3)/((b2-b1)*r^2*ksei^2) * u/(2);
            case 2
	            value1 = (sigma_h2-r*ksei^2)/(r^2*ksei^2) * u/(2);
		        dify = 1/sqrt(2*pi)*exp(-1/2*u ^2)* (-u);	
                dify1 = (sigma_h2-r*ksei^2)/(r^2*ksei^2) /(2);
                value2 = ((sigma_h2-r*ksei^2)/(r^2*ksei^2))^2 * (u^3-3*u)/(factorial(2*2));
                value = (value1*value1*dify/2-value2-value1*dify1)/(1+normpdf(u)*value1);
            case 3
                psi1 = g_tuda_esti(u, ksei, g2,g3,sigma_h2, b2,b1,1,randomness,r);
                psi2 = g_tuda_esti(u, ksei, g2,g3,sigma_h2, b2,b1,2,randomness,r);
		        syms z 
		        f = 1/sqrt(2*pi)*exp(-1/2*z ^2);
		        dify = diff(f);	
                dify2 = diff(dify);
                gamma1 = (sigma_h2-r*ksei^2)/(r^2*ksei^2) * z /(2);
                difg1 = diff(gamma1);
                gamma2 = ((sigma_h2-r*ksei^2)/(r^2*ksei^2))^2 * (z ^3-3*z)/(factorial(2*2));
                difg2 = diff(gamma2);
                gamma3 = ((sigma_h2-r*ksei^2)/(r^2*ksei^2))^3 * (u ^5-10*u^3+15*u)/(factorial(2*3));
                syms vs
                eqn = vs == psi1*psi2*vpa(subs(dify,z,u))/2+psi2*psi1*vpa(subs(dify,z,u))/2+psi1^3*vpa(subs(dify2,z,u))/factorial(3)...
                            -(gamma3+psi2*vpa(subs(difg1,z,u))+psi1*vpa(subs(difg2,z,u)))-(vs*normpdf(u))*(vpa(subs(gamma2,z,u))+psi2*vpa(subs(difg1,z,u)))...
                                -(vs*normpdf(u)+psi1*psi2*vpa(subs(dify,z,u))/2+psi1*psi2*vpa(subs(dify,z,u))/2)*vpa(subs(gamma1,z,u));
                value = solve(eqn,vs);

        end
    case 2
        switch l
            case 1
                value = 0;
            case 2
                value = 0;
            case 3
                value = 0;
        end
    end

end
		
	