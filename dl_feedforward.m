function u_dl = dl_feedforward(feat_window, welm_model, dnn_net, ...
                                X_mu, X_sig, Y_mu, Y_sig, itae_w, itae_d)
    x_raw  = feat_window(:)';
    x_norm = (x_raw - X_mu) ./ X_sig;
    pred_w = welm_predict(welm_model, x_norm);   % [1 × 2]
    pred_d = dnn_predict(dnn_net,     x_norm);   % [1 × 2]
    % Denormalise
    pw = pred_w .* Y_sig + Y_mu;   % [1 × 2]
    pd = pred_d .* Y_sig + Y_mu;   % [1 × 2]
    % Confidence weighting
    cw=1/(itae_w+1e-10); cd=1/(itae_d+1e-10);
    aw=cw/(cw+cd); ad=cd/(cw+cd);
    u_vec = aw*pw + ad*pd;         % [1 × 2]
    u_dl  = mean(u_vec);           % scalar fed into control law
end

