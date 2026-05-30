function u_a = mianfis(e,de,p)
    ce=p(1:3); se=abs(p(4:6))+0.01; cd=p(7:9); sd=abs(p(10:12))+0.01;
    ar=p(13:21); br=p(22:30); cr=p(31:39);
    me=gmf(e,ce,se); md=gmf(de,cd,sd);
    k=0; w=zeros(1,9);
    for i=1:3; for j=1:3; k=k+1; w(k)=me(i)*md(j); end; end
    wb=w/(sum(w)+1e-10); f_r=ar*e+br*de+cr; u_a=dot(wb,f_r);
end