function w = gl_weights(alpha, M)
    w = zeros(1,M);  w(1) = 1;
    for k = 2:M
        w(k) = w(k-1) * (k - 1 - alpha) / (k - 1);
    end
end