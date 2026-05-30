%% =========================================================================
function [X_feat, Y_tgt, t_out, dt_out] = load_simulink_data(fname, seq_len, n_inputs, n_outputs)

    data = load(fname);
    out  = data.out;   
    % Extract time axis from deltaF (reference signal)
    t_raw  = out.deltaF.Time;
    dt_out = t_raw(2) - t_raw(1);
    N_raw  = length(t_raw);
    t_out  = t_raw;

    resamp = @(sig) interp1(linspace(0,1,length(sig)), sig, linspace(0,1,N_raw));

    % Compute RoCoF from deltaF signals
    df1_raw   = out.deltaF.Data;
    df2_raw   = out.deltaF2.Data;
    rocof1    = [0; diff(df1_raw) / dt_out];
    rocof2    = [0; diff(df2_raw) / dt_out];

    % Assemble all 15 columns — same order as your dataset_ script
    X_raw = [resamp(out.BESS1.Data(:))',       ...  % 1
             resamp(out.BESS2.Data(:))',        ...  % 2
             resamp(out.DISEL1.Data(:))',       ...  % 3
             resamp(out.DISEL2.Data(:))',       ...  % 4
             resamp(out.PPV1.Data(:))',         ...  % 5
             resamp(out.PPV2.Data(:))',         ...  % 6
             resamp(out.WIND1.Data(:))',        ...  % 7
             resamp(out.WIND2.Data(:))',        ...  % 8
             resamp(df1_raw(:))',               ...  % 9  deltaF
             resamp(df2_raw(:))',               ...  % 10 deltaF2
             resamp(out.deltademand1.Data(:))', ...  % 11
             resamp(out.deltademand2.Data(:))', ...  % 12
             resamp(out.ptie_error.Data(:))',   ...  % 13
             resamp(rocof1(:))',                ...  % 14 RoCoF area1
             resamp(rocof2(:))'];                    % 15 RoCoF area2


    target1_raw = resamp(out.deltademand1.Data(:))' ...
                - resamp(out.PPV1.Data(:))'          ...
                - resamp(out.WIND1.Data(:))';

    target2_raw = resamp(out.deltademand2.Data(:))' ...
                - resamp(out.PPV2.Data(:))'          ...
                - resamp(out.WIND2.Data(:))';

    Y_raw = [target1_raw, target2_raw];   % [N_raw × 2]
    assert(size(X_raw,2) == n_inputs,  ...
        'Feature count mismatch: expected %d, got %d', n_inputs, size(X_raw,2));
    assert(size(Y_raw,2) == n_outputs, ...
        'Output count mismatch: expected %d, got %d', n_outputs, size(Y_raw,2));


    N_samples = N_raw - seq_len;
    X_feat    = zeros(N_samples, seq_len * n_inputs);
    Y_tgt     = zeros(N_samples, n_outputs);

    for k = 1:N_samples
        window          = X_raw(k : k+seq_len-1, :);   % [seq_len × n_inputs]
        X_feat(k, :)    = window(:)';                   % flatten row-major
        Y_tgt(k, :)     = Y_raw(k+seq_len, :);         % target at next step
    end

    fprintf('   Dataset loaded: %d raw timesteps → %d sequence samples\n', N_raw, N_samples);
    fprintf('   Features: %d  |  Targets: %d\n', n_inputs, n_outputs);
end