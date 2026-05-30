function [OS, Ts, Tr, MD] = pmetrics(t, y)

    % Max deviation (peak absolute excursion)
    MD = max(abs(y));

    % Steady-state: mean of last 100 samples
    fv = mean(y(end - min(100, length(y)-1) : end));

    % --- Overshoot ---
    % Find the primary peak (first large excursion after disturbance)
    [~, i_peak] = max(abs(y));

    % Look for a secondary crossing on the opposite side after the peak
    % (true overshoot = signal crosses past fv in the opposite direction)
    y_after = y(i_peak:end);
    sign_peak = sign(y(i_peak));

    % Samples after peak where signal crosses to opposite side of fv
    opposite = y_after - fv;
    cross_idx = find(sign_peak * opposite < 0, 1);  % first sign reversal past fv

    if ~isempty(cross_idx)
        % True overshoot: max excursion on the rebound side
        y_rebound = y(i_peak + cross_idx - 1 : end);
        OS = max(abs(y_rebound - fv));
    else
        % No crossing — underdamped or critically damped, OS near zero
        OS = max(0, sign_peak * (fv - y(i_peak)));  % 0 if no rebound
        OS = 0;  % conservative: no crossing = no overshoot
    end

    % --- Settling time (2% of MD band around fv) ---
    band = 0.02 * MD;
    i_settle = find(abs(y - fv) > band, 1, 'last');
    if isempty(i_settle)
        Ts = 0;
    else
        Ts = t(i_settle);
    end

    % --- Rise time (10% to 90% of peak) ---
    p = max(abs(y));
    i1 = find(abs(y) >= 0.10 * p, 1);
    i9 = find(abs(y) >= 0.90 * p, 1);
    if isempty(i1) || isempty(i9)
        Tr = NaN;
    else
        Tr = t(i9) - t(i1);
    end
end