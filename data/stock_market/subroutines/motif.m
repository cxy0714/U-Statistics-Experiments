function [result_f]= motif(data1,data2,type,list1)
    result_f = zeros(size(list1,1),1);
    orderli = type.permu;
    for i = 1:size(list1,1)
        xx1 = data1(list1(i,1),:);
        xx2 = data1(list1(i,2),:);
        xx3 = data1(list1(i,3),:);
        xx4 = data1(list1(i,4),:);
        xx = [xx1;xx2;xx3;xx4];
        yy1 = data2(list1(i,1),:);
        yy2 = data2(list1(i,2),:);
        yy3 = data2(list1(i,3),:);
        yy4 = data2(list1(i,4),:);
        yy = [yy1;yy2;yy3;yy4];       
        switch type.name
            case 'stockmarket'
                distallx = zeros(4,4);
                for k = 1:4
                    for l = (k+1):4
                        distallx(k,l) = norm(xx(k,:)-xx(l,:));
                        distallx(l,k) = distallx(k,l);
                    end
                end
                distally = zeros(4,4);
                for k = 1:4
                    for l = (k+1):4
                        distally(k,l) = norm(yy(k,:)-yy(l,:)); 
                        distally(l,k) = distally(k,l);
                    end
                end
                disli = zeros(24,1);
                for m = 1:24
                    orderli1 = orderli(m,:);
                    s = orderli1(1);
                    t = orderli1(2);
                    u = orderli1(3);
                    v = orderli1(4);
                    disli(m,1) = distallx(s,t)*distally(s,t)+distallx(s,t)*distally(u,v)...
                                    -2*distallx(s,t)*distally(s,u);
                end
                f = 1/24*sum(disli);
        end
        result_f(i,1) = f;
    end
end

