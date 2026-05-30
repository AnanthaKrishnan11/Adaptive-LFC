function [df1, df2, dPtie_out, u_out, u_dl_out] = run_simulation(t, dt, Ad, Bd, ctrl_type, ...
    p_pi, p_tfoid, p_anfis, alpha_w, beta_w, gamma_w, dPL1, dPL2, M_gl, ...
    welm_model, dnn_net, X_mu, X_sig, Y_mu, Y_sig, seq_len, n_inputs)

    Nt = length(t);

    % State vectors
    x1 = zeros(5,1);
    x2 = zeros(5,1);

    % Output arrays
    df1      = zeros(Nt,1);
    df2      = zeros(Nt,1);
    dPtie_out = zeros(Nt,1);
    u_out    = zeros(Nt,2);
    u_dl_out = zeros(Nt,1);

    % PI integrator states
    xi1 = 0;
    xi2 = 0;

    % GL error history buffers  (length = M_gl, newest at index 1)
    e1_h = zeros(M_gl, 1);
    e2_h = zeros(M_gl, 1);

    % Feature buffers for DL feedforward
    fbuf1 = zeros(seq_len, n_inputs);
    fbuf2 = zeros(seq_len, n_inputs);

    % Confidence weights for WELM vs DNN blending
    itae_welm = 1.0;
    itae_dnn  = 1.0;
    tau_itae  = 200;   % steps between ITAE weight updates

    % PI gains (extracted once)
    Kp_pi = p_pi(1);
    Ki_pi = p_pi(2);

    % FIX 5 — Low-pass filter states for derivative signals fed to ANFIS
    de1_f     = 0;
    de2_f     = 0;
    alpha_de  = 0.08;   % smoothing coefficient  (tune 0.05–0.15)

    % ------------------------------------------------------------------ %
    %  MAIN SIMULATION LOOP                                               %
    % ------------------------------------------------------------------ %
    for k = 1:Nt

        % Record outputs at current step
        df1(k)       = x1(1);
        df2(k)       = x2(1);
        dPtie_out(k) = x1(5);

        % ---- Error signals ----
        e1 = -x1(1);
        e2 = -x2(1);

        % ---- Raw one-step derivative (used internally + for feature vector) ----
        de1_raw = (e1 - e1_h(1)) / dt;
        de2_raw = (e2 - e2_h(1)) / dt;

        % FIX 5 — Smoothed derivative for ANFIS (avoids high-frequency noise)
        de1_f = alpha_de * de1_raw + (1 - alpha_de) * de1_f;
        de2_f = alpha_de * de2_raw + (1 - alpha_de) * de2_f;

        % ---- Update GL error history (shift right, insert new at front) ----
        e1_h = [e1; e1_h(1:end-1)];
        e2_h = [e2; e2_h(1:end-1)];

        % ---- Auxiliary signals for feature vector ----
        rocof1  = de1_raw;                             % RoCoF uses raw dF/dt
        soc     = 50 + 10 * sin(2*pi*0.02 * k*dt);    % synthetic SOC
        pv_f    = 0.3 + 0.1 * sin(2*pi*0.05 * k*dt);
        wnd_f   = 0.2 + 0.1 * sin(2*pi*0.08 * k*dt);

        % ---- 15-feature vectors (must match dataset column order) ----
        feat1 = [x1(1), rocof1,   x1(5), pv_f, wnd_f, dPL1(k), max(0, x1(2)), soc/100, ...
                 x1(3), x1(4),    dPL1(k)*0.5, dPL2(k)*0.5, x1(5)-x2(5), de1_raw, de2_raw];
        feat2 = [x2(1), de2_raw,  x2(5), pv_f, wnd_f, dPL2(k), max(0, x2(2)), soc/100, ...
                 x2(3), x2(4),    dPL1(k)*0.5, dPL2(k)*0.5, x1(5)-x2(5), de1_raw, de2_raw];

        % Shift feature buffers (newest row at top)
        fbuf1 = [feat1; fbuf1(1:end-1, :)];
        fbuf2 = [feat2; fbuf2(1:end-1, :)];

        % ================================================================
        %  LAYER 1 — Base controller  (PI  or  TFOID)
        % ================================================================
        switch ctrl_type
            case 'PI'
                xi1   = xi1 + e1 * dt;
                xi2   = xi2 + e2 * dt;
                u1_tf = Kp_pi * e1 + Ki_pi * xi1;
                u2_tf = Kp_pi * e2 + Ki_pi * xi2;

            case {'TFOID', 'HYBRID'}
                % FIX 1 + 2 applied inside tfoid_ctrl (see corrected function)
                u1_tf = tfoid_ctrl(e1_h, p_tfoid, dt, M_gl);
                u2_tf = tfoid_ctrl(e2_h, p_tfoid, dt, M_gl);
        end

        % ================================================================
        %  LAYER 2 — ANFIS adaptive correction
        % ================================================================
        u1_an = 0;
        u2_an = 0;
        if strcmp(ctrl_type, 'HYBRID')
            % FIX 5 — pass smoothed derivative instead of raw
            u1_an = mianfis(e1, de1_f, p_anfis);
            u2_an = mianfis(e2, de2_f, p_anfis);
        end

        % ================================================================
        %  LAYER 3 — WELM-DNN feedforward compensation
        % ================================================================
        u1_dl = 0;
        if strcmp(ctrl_type, 'HYBRID')
            u1_dl = dl_feedforward(fbuf1, welm_model, dnn_net, ...
                                   X_mu, X_sig, Y_mu, Y_sig, ...
                                   itae_welm, itae_dnn);

            % Periodically update ITAE-based confidence weights
            if mod(k, tau_itae) == 0 && k > tau_itae
                win      = max(1, k - tau_itae) : k;
                err_win  = df1(win);
                t_win    = t(win) - t(win(1));
                itae_welm = trapz(t_win, t_win .* abs(err_win)) + 1e-6;
                itae_dnn  = itae_welm * (0.9 + 0.2*rand);
            end
        end
        u_dl_out(k) = u1_dl;

        % ================================================================
        %  FUSE all three layers with optimised mixing weights
        % ================================================================
        u1 = alpha_w * u1_tf + beta_w * u1_an + gamma_w * u1_dl;
        u2 = alpha_w * u2_tf + beta_w * u2_an + gamma_w * u1_dl;

        % Actuator saturation
        u1 = max(min(u1,  1.5), -1.5);
        u2 = max(min(u2,  1.5), -1.5);
        u_out(k, :) = [u1, u2];

        % ================================================================
        %  DISCRETE STATE UPDATE  x[k+1] = Ad·x[k] + Bd·u[k]
        % ================================================================
        x1 = Ad * x1 + Bd * [u1;  u1*0.3;  dPL1(k)];
        x2 = Ad * x2 + Bd * [u2;  u2*0.3;  dPL2(k)];

        % Tie-line power update (Euler integration of ACE)
        x1(5) = x1(5) + dt * 2*pi * 0.0866 * x1(1);
        x2(5) = x2(5) + dt * 2*pi * 0.0866 * x2(1);

    end  % end main loop
end



