function y = gmf(x,c,s); y=exp(-0.5*((x-c)./max(s,1e-6)).^2); end
