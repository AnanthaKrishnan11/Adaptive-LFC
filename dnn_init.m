

function net = dnn_init(n_in, layer_sizes, n_out, seed)
    if nargin<4, seed=1; end
    rng(seed);
    sizes   = [n_in, layer_sizes, n_out];   % <-- n_out instead of hardcoded 1
    n_layer = length(sizes)-1;
    net.W   = cell(n_layer,1);
    net.b   = cell(n_layer,1);
    net.mW  = cell(n_layer,1);
    net.vW  = cell(n_layer,1);
    net.mb  = cell(n_layer,1);
    net.vb  = cell(n_layer,1);
    for l = 1:n_layer
        sc        = sqrt(2/sizes(l));
        net.W{l}  = randn(sizes(l), sizes(l+1)) * sc;
        net.b{l}  = zeros(1, sizes(l+1));
        net.mW{l} = zeros(size(net.W{l}));
        net.vW{l} = zeros(size(net.W{l}));
        net.mb{l} = zeros(size(net.b{l}));
        net.vb{l} = zeros(size(net.b{l}));
    end
    net.n_layer = n_layer;
    net.t_adam  = 0;
end
