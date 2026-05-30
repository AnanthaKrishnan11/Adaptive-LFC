function model = welm_train(X_tr, Y_tr, N_h, seed)
    if nargin<4, seed=0; end
    rng(seed);
    n_feat = size(X_tr,2);
    Win    = randn(n_feat, N_h) * 0.5;
    b_in   = randn(1, N_h)     * 0.1;
    H_mat  = relu(X_tr * Win + repmat(b_in, size(X_tr,1), 1));
    lam_r  = 1e-4;
    % beta : [N_h × n_outputs]  (works for any number of outputs)
    beta   = (H_mat'*H_mat + lam_r*eye(N_h)) \ (H_mat' * Y_tr);
    model.Win  = Win;
    model.b_in = b_in;
    model.beta = beta;
    model.N_h  = N_h;
end