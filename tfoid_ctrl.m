function u = tfoid_ctrl(e_hist, p, dt, M)
    Kp=p(1); Ki=p(2); lam=p(3); Kd=p(4); mu=p(5); sig=p(6);
    N = min(numel(e_hist), M);  ew = e_hist(1:N)';
    w_i = gl_weights(-lam, N);
    w_d = gl_weights( mu,  N);
    w_t = gl_weights(-sig, N);
    fi = (dt^ lam)  * (w_i * ew');   % fractional integral
    fd = (dt^(-mu)) * (w_d * ew');   % fractional derivative
    ft = (dt^(-sig))* (w_t * ew');   % tilt — was dt^sig (wrong!), now dt^(-sig)
    u  = Kp*e_hist(1) + Ki*fi + Kd*fd + ft;
end