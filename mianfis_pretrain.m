function [p_out, loss_hist] = mianfis_pretrain(e_tr, de_tr, y_tr, opts)

    lr     = opts.lr;       % 0.01  (Table 7 / Section 3.6)
    epochs = opts.epochs;   % 500   (Table 7)
    n_mf   = opts.n_mf;     % 3     (Section 3.6: Low/Medium/High)
    N      = length(e_tr);

    % ---------------------------------------------------------------
    % Initialise MF centres: evenly spaced across data range (Eq. 32)
    % ---------------------------------------------------------------
    e_range  = linspace(min(e_tr),  max(e_tr),  n_mf);   % [1×3]
    de_range = linspace(min(de_tr), max(de_tr), n_mf);   % [1×3]

    % Spreads: set to half the spacing between centres (standard init)
    if n_mf > 1
        sig_e  = repmat(abs(e_range(2)  - e_range(1))  / 2 + 1e-3, 1, n_mf);
        sig_de = repmat(abs(de_range(2) - de_range(1)) / 2 + 1e-3, 1, n_mf);
    else
        sig_e  = ones(1, n_mf);
        sig_de = ones(1, n_mf);
    end

    n_rules = n_mf * n_mf;   % 9

    loss_hist = zeros(epochs, 1);

    for ep = 1:epochs

        % -----------------------------------------------------------
        % FORWARD PASS: compute rule firing strengths for all samples
        % -----------------------------------------------------------
        % Gaussian MF values  [N × n_mf]  (Eq. 32 / Eq. 38)
        MF_e  = exp(-0.5 * ((e_tr  - e_range)  ./ max(sig_e,  1e-6)).^2);   % [N×3]
        MF_de = exp(-0.5 * ((de_tr - de_range) ./ max(sig_de, 1e-6)).^2);   % [N×3]

        % Rule firing strengths w_{ij} = mu_e_i * mu_de_j  [N × 9]
        W = zeros(N, n_rules);
        r = 0;
        for i = 1:n_mf
            for j = 1:n_mf
                r = r + 1;
                W(:, r) = MF_e(:, i) .* MF_de(:, j);
            end
        end

        % Normalised firing strengths  [N × 9]  (Eq. 39 denominator)
        W_sum  = sum(W, 2) + 1e-10;          % [N×1]
        W_bar  = W ./ W_sum;                  % [N×9]

        % -----------------------------------------------------------
        % PHASE 1 — LSE for consequent parameters (Eq. 34)
        % Each rule output: f_r = a_r*e + b_r*de + c_r
        % Build regressor matrix Phi [N × 27]:
        %   each rule contributes 3 columns: w_bar_r*e, w_bar_r*de, w_bar_r
        % -----------------------------------------------------------
        Phi = zeros(N, n_rules * 3);
        for r = 1:n_rules
            Phi(:, (r-1)*3+1) = W_bar(:, r) .* e_tr;
            Phi(:, (r-1)*3+2) = W_bar(:, r) .* de_tr;
            Phi(:, (r-1)*3+3) = W_bar(:, r);
        end

        % Least-squares solution (ridge-regularised for stability)
        lam_ls  = 1e-4;
        cons_vec = (Phi' * Phi + lam_ls * eye(n_rules*3)) \ (Phi' * y_tr);  % [27×1]

        % Predicted output and MSE
        y_hat = Phi * cons_vec;                    % [N×1]
        err   = y_hat - y_tr;                      % [N×1]
        loss_hist(ep) = mean(err.^2);

        % -----------------------------------------------------------
        % PHASE 2 — Gradient Descent for antecedent params (Eq. 35)
        % Backprop through W_bar → W → MF_e, MF_de
        % dL/d(c_i) and dL/d(sigma_i) for each MF centre and spread
        % -----------------------------------------------------------
        % Reshape consequents: [9 × 3]  (a_r, b_r, c_r per rule)
        cons_mat = reshape(cons_vec, 3, n_rules)';   % [9×3]

        % Rule outputs  [N×9]
        F_r = W_bar .* (e_tr .* cons_mat(:,1)' + de_tr .* cons_mat(:,2)' + cons_mat(:,3)');

        % dL/d(W_bar_r)  [N×9]
        f_r_vals = e_tr .* cons_mat(:,1)' + de_tr .* cons_mat(:,2)' + cons_mat(:,3)';
        dL_dWbar = (2/N) .* err .* f_r_vals;         % [N×9]

        % dL/d(W_r) via normalisation:  dWbar_r/dW_r = (W_sum - W_r)/W_sum^2
        dL_dW = (dL_dWbar .* W_sum - sum(dL_dWbar .* W, 2)) ./ (W_sum.^2);  % [N×9]

        % Accumulate gradients for MF parameters
        d_ce   = zeros(1, n_mf);
        d_se   = zeros(1, n_mf);
        d_cde  = zeros(1, n_mf);
        d_sde  = zeros(1, n_mf);

        r = 0;
        for i = 1:n_mf
            for j = 1:n_mf
                r = r + 1;
                % dW_r / d(mu_e_i)  = mu_de_j
                % dW_r / d(mu_de_j) = mu_e_i
                dL_dMFe_i  = dL_dW(:, r) .* MF_de(:, j);   % [N×1]
                dL_dMFde_j = dL_dW(:, r) .* MF_e(:, i);    % [N×1]

                % dMF/dc  = MF * (x - c) / sigma^2
                % dMF/ds  = MF * (x - c)^2 / sigma^3
                diff_e  = e_tr  - e_range(i);
                diff_de = de_tr - de_range(j);

                d_ce(i)  = d_ce(i)  + sum(dL_dMFe_i  .* MF_e(:,i)  .* diff_e  / max(sig_e(i),1e-6)^2);
                d_se(i)  = d_se(i)  + sum(dL_dMFe_i  .* MF_e(:,i)  .* diff_e.^2 / max(sig_e(i),1e-6)^3);
                d_cde(j) = d_cde(j) + sum(dL_dMFde_j .* MF_de(:,j) .* diff_de / max(sig_de(j),1e-6)^2);
                d_sde(j) = d_sde(j) + sum(dL_dMFde_j .* MF_de(:,j) .* diff_de.^2 / max(sig_de(j),1e-6)^3);
            end
        end

        % Scale by 2/N (MSE gradient factor)
        d_ce  = (2/N) * d_ce;
        d_se  = (2/N) * d_se;
        d_cde = (2/N) * d_cde;
        d_sde = (2/N) * d_sde;

        % Gradient descent update (Eq. 35)
        e_range  = e_range  - lr * d_ce;
        sig_e    = abs(sig_e  - lr * d_se)  + 1e-6;   % keep positive
        de_range = de_range - lr * d_cde;
        sig_de   = abs(sig_de - lr * d_sde) + 1e-6;   % keep positive

        if mod(ep, 100) == 0
            fprintf('     ANFIS pre-train epoch %3d / %d  |  MSE = %.6f\n', ...
                    ep, epochs, loss_hist(ep));
        end
    end


    p_out = [e_range, de_range, sig_e, sig_de, cons_vec'];

    fprintf('   ANFIS pre-training done. Param vector [1×%d] assembled.\n', length(p_out));
end

