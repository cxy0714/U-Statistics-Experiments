function [Ir_list,output_check_OK] = randomsamle(n,alpha)
    N = floor(n^alpha);
    output_check_OK = false;
    raw_x = rand(10*N,1);
    raw_x = floor(raw_x*n^4);
    i1_list = floor(raw_x/n^3);
    raw_y = raw_x - i1_list*n^3;
    i2_list = floor(raw_y/n^2);
    raw_z = raw_y - i2_list*n^2;
    i3_list = floor(raw_z/n);
    raw_k = raw_z - i3_list*n;
    i4_list = floor(raw_k);
    keep_elem = ones(size(i1_list));
    keep_elem(i1_list==i2_list) = 0;
    keep_elem(i1_list==i3_list) = 0;
    keep_elem(i2_list==i3_list) = 0;
    keep_elem(i1_list==i4_list) = 0;
    keep_elem(i2_list==i4_list) = 0;
    keep_elem(i3_list==i4_list) = 0;
    Ir_list = [i1_list, i2_list, i3_list, i4_list];
    Ir_list = Ir_list(keep_elem>0.5,:);
    if(size(Ir_list,1)>=N)
        output_check_OK = true;
        Ir_list = Ir_list(1:N,:)+1;
    end 

end

