function d=stoch(t,amp,seed)
    rng(seed);
    d=amp*(0.5*sin(2*pi*0.3*t)+0.3*randn(length(t),1)+0.2*sin(2*pi*0.8*t));
    d=d.*(t>=1);
end
