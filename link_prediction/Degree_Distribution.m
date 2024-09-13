function [DeD,aver_DeD] = Degree_Distribution(A)
N=size(A,2);
DeD=zeros(1,N); 
for i=1:N 
   % DeD(i)=length(find((A(i,:)==1)));
    DeD(i)=sum(A(i,:));
end 
aver_DeD=mean(DeD);
end

