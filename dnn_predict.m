function y = dnn_predict(net, X)
    A=X;
    for l=1:net.n_layer
        Z=A*net.W{l}+repmat(net.b{l},size(A,1),1);
        if l<net.n_layer; A=relu(Z); else; A=Z; end
    end
    y=A;
end