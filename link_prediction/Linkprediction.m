function similarity = Linkprediction(V, N)

[DeD,aver_DeD]=Degree_Distribution(V);%degree of graph
R=zeros(N,N);%
for i=1:N
    R(i,:)=V(i,:)/DeD(i);
end
R1=R';
h=find(V==1);%connected links
linknumber=length(h);%num of links
    Q=eye(N,N);%Q = pi_xy(t)
    S=zeros(N,100);
    for t=1:200 %t is big enough
    for i=1:N % similarity
        Q(:,i)=R1*Q(:,i); 
    for j=1:(N-1)
        K1=DeD(j);%kx
        K2=K1/linknumber;%kx/2M
        K3=DeD(N);%degree of ky
        K4=K3/linknumber;%ky/2M   
        P1=Q(j,N);%pixy(i)
        P2=Q(N,j);%piyx(i)
        S1=K2*P1+K4*P2;%S(LRW)
        S(j,t)=S1;
    end
    end
    S2=sum(S,2);%S(SRW)
    similarity=find(S2==max(S2)); 
end