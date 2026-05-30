clear; clc; close all;
rng(42);

% Params initialization
% Simulation
Tsim  = 20;
dt    = 1e-3;
t     = (0:dt:Tsim)';
Nt    = length(t);

% % CBES  (fast mode)
% N_pop  = 15;
% T_max  = 20;
% N_runs = 1;
% TO run fully comment out below
N_pop  = 30;
T_max  = 100;
N_runs = 20;
% Deep learning
N_welm    = 50;          % WELM hidden neurons
N_dnn     = [100,50,25]; % DNN layer sizes
seq_len   = 20;          % input window length (time steps)
n_inputs  = 15;          
n_outputs = 2;           

% GL fractional memory
M_gl   = 800;

%  SYSTEM PARAMETERS

H    = 0.1667;   D    = 0.015;   R    = 3.0;
T12  = 0.0866;   Tg   = 0.4;     Tt   = 0.08;
T_bess = 0.1;    K_bess = 1.0;
dPL_step  = 0.10;
dPL_large = 0.20;
noise_amp = 0.05;

%  DISCRETE STATE-SPACE MODEL  (5 states per area)
%  x = [df, dPm, dPv, dPbess, dPtie]

A_c = [ -D/(2*H),  1/(2*H),  0,         1/(2*H),  -1/(2*H) ;
         0,        -1/Tt,     1/Tt,      0,         0        ;
        -1/(R*Tg),  0,        -1/Tg,     0,         0        ;
         0,         0,         0,        -1/T_bess,  0        ;
         2*pi*T12,  0,         0,         0,         0        ];

B_c = [  0,              0,           -1/(2*H) ;
          0,              0,            0       ;
          1/Tg,           0,            0       ;
          0,              K_bess/T_bess, 0      ;
          0,              0,            0       ];

Ad = eye(5) + dt*A_c;
Bd = dt*B_c;

%   LOAD   DATA


fprintf('>>> Step 1/5 : Loading Simulink dataset...\n');

[X_all, Y_all, t_data, dt_data] = load_simulink_data('dataset.mat', seq_len, n_inputs, n_outputs);

n_synth = size(X_all, 1);   % actual number of samples from dataset
fprintf('   Total sequence samples extracted: %d\n', n_synth);

% Normalise features
X_mu  = mean(X_all, 1);
X_sig = std(X_all, 0, 1) + 1e-8;
X_norm = (X_all - X_mu) ./ X_sig;

Y_mu  = mean(Y_all, 1);
Y_sig = std(Y_all, 0, 1) + 1e-8;
Y_norm = (Y_all - Y_mu) ./ Y_sig;

% Train / Val / Test  split  70/15/15
n_tr  = round(0.70 * n_synth);
n_val = round(0.15 * n_synth);
X_tr  = X_norm(1:n_tr, :);
Y_tr  = Y_norm(1:n_tr, :);
X_val = X_norm(n_tr+1 : n_tr+n_val, :);
Y_val = Y_norm(n_tr+1 : n_tr+n_val, :);
X_te  = X_norm(n_tr+n_val+1 : end, :);
Y_te  = Y_norm(n_tr+n_val+1 : end, :);

fprintf('   Training: %d  |  Validation: %d  |  Test: %d samples\n', ...
        size(X_tr,1), size(X_val,1), size(X_te,1));
fprintf('   Feature dimension: %d  (seq_len=%d × n_inputs=%d)\n', ...
        seq_len*n_inputs, seq_len, n_inputs);
fprintf('   Output dimension : %d  (target1, target2)\n', n_outputs);

%   TRAIN WELM

fprintf('\n Step 2/5 : Training WELM (%d hidden neurons)\n', N_welm);

welm_model = welm_train(X_tr, Y_tr, N_welm, 42);

y_welm_te = welm_predict(welm_model, X_te);
welm_mse  = mean(mean((y_welm_te - Y_te).^2));
fprintf('   WELM Test MSE (normalised, mean over outputs): %.6f\n', welm_mse);

%  TRAIN DNN 


fprintf('\n>>> Step 3/5 : Training DNN ...\n');

n_feat_total = seq_len * n_inputs;
dnn_net = dnn_init(n_feat_total, N_dnn, n_outputs, 1);  % <-- n_outputs=2

dnn_opts.lr       = 0.001;
dnn_opts.epochs   = 200;
dnn_opts.batch    = 32;
dnn_opts.l2       = 1e-4;
dnn_opts.dropout  = 0.1;
dnn_opts.patience = 20;

[dnn_net, dnn_loss] = dnn_train(dnn_net, X_tr, Y_tr, X_val, Y_val, dnn_opts);

y_dnn_te = dnn_predict(dnn_net, X_te);
dnn_mse  = mean(mean((y_dnn_te - Y_te).^2));
fprintf('   DNN  Test MSE (normalised, mean over outputs): %.6f\n', dnn_mse);



fprintf('\n>>> Step 3.5/5 : MIANFIS Pre-Training on Simulink data ...\n');
fprintf('   Method : LSE (consequents) + Gradient Descent (antecedents)\n');
fprintf('   Epochs : 500  |  Learning rate : 0.01  (Table 7 / Section 3.6)\n');


col_df1   = seq_len * 9;          
col_rocof = seq_len * 14;          

e_tr  = -X_tr(:, col_df1);        % frequency error  e = -deltaF
de_tr =  X_tr(:, col_rocof);      % smoothed RoCoF ≈ de/dt (already normalised)

% Target: normalised net disturbance area 1  (what ANFIS should correct)
y_anfis_tr = Y_tr(:, 1);           % [n_tr x 1]

% Pre-train ANFIS using the function defined at the bottom of this file
anfis_pretrain_opts.lr     = 0.01;   % Table 7: learning rate
anfis_pretrain_opts.epochs = 500;    % Table 7: training epochs
anfis_pretrain_opts.n_mf   = 3;      % 3 Gaussian MFs per input (Section 3.6)

[p_anfis_pretrained, anfis_pretrain_loss] = ...
    mianfis_pretrain(e_tr, de_tr, y_anfis_tr, anfis_pretrain_opts);

fprintf('   Pre-training complete. Final MSE = %.6f\n', anfis_pretrain_loss(end));

% =========================================================================
%  PART 12 : CBES OPTIMISATION
%  (unchanged — only warm-start injection added after population init)
% =========================================================================

fprintf('\n CBES Optimisation...\n');
fprintf('   Population=%d  Iterations=%d  Runs=%d\n', N_pop, T_max, N_runs);

n_var   = 48;
lb_tf = [0.0, 0.0, 0.3, 0.0, 0.3, 0.1];   % Kp, Ki, lambda, Kd, mu, sigma
ub_tf = [3.0, 3.0, 1.2, 3.0, 1.2, 1.2];  
lb_an   = [repmat(-2,1,12), repmat(-1,1,27)];
ub_an   = [repmat( 2,1,12), repmat( 1,1,27)];
lb_w    = [0.1,0.0,0.0];
ub_w    = [1.0,1.0,0.5];
lb = [lb_tf, lb_an, lb_w];
ub = [ub_tf, ub_an, ub_w];

dPL1_o       = zeros(Nt,1);  dPL1_o(t>=1) = dPL_step;
dPL2_o       = zeros(Nt,1);  dPL2_o(t>=2) = dPL_step*0.8;

pa=0.25; beta_lv=1.5; f_min=0; f_max=2;
alpha_bat=0.9; gamma_bat=0.9;

best_global        = inf;
best_params_global = lb + rand(1,n_var).*(ub-lb);
conv_all           = zeros(N_runs, T_max);

for run = 1:N_runs
    pop  = lb + rand(N_pop,n_var).*(ub-lb);
    vel  = zeros(N_pop,n_var);
    A_b  = ones(N_pop,1);
    r_b  = 0.5*ones(N_pop,1);
    fit  = zeros(N_pop,1);
    p_anfis_clipped = min(max(p_anfis_pretrained, lb_an), ub_an);
    pop(1, 7:45)    = p_anfis_clipped;
    fprintf('    ANFIS params set from pre-training.\n');

    for i = 1:N_pop
        fit(i) = cbes_fitness(pop(i,:),t,dt,Ad,Bd,dPL1_o,dPL2_o,M_gl,...
                              welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,...
                              seq_len,n_inputs);
    end
    [bf,bi] = min(fit);  Xb = pop(bi,:);
    cv = zeros(1,T_max);

    for iter = 1:T_max
        np = pop;
        for i = 1:N_pop
            su = (gamma(1+beta_lv)*sin(pi*beta_lv/2) / ...
                 (gamma((1+beta_lv)/2)*beta_lv*2^((beta_lv-1)/2)))^(1/beta_lv);
            ul = randn(1,n_var)*su;  vl = randn(1,n_var);
            step = ul ./ (abs(vl).^(1/beta_lv));
            Xcs  = pop(i,:) + 0.01*step.*(pop(i,:)-Xb);
            Xcs  = min(max(Xcs,lb),ub);
            fcs  = cbes_fitness(Xcs,t,dt,Ad,Bd,dPL1_o,dPL2_o,M_gl,...
                                welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,...
                                seq_len,n_inputs);

            fi_b = f_min + (f_max-f_min)*rand;
            vel(i,:) = vel(i,:) + (pop(i,:)-Xb)*fi_b;
            Xbt = pop(i,:) + vel(i,:);
            if rand > r_b(i)
                Xbt = Xb + 0.001*randn(1,n_var);
            end
            Xbt = min(max(Xbt,lb),ub);
            fbt = cbes_fitness(Xbt,t,dt,Ad,Bd,dPL1_o,dPL2_o,M_gl,...
                               welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,...
                               seq_len,n_inputs);

            if fcs < fbt; Xnew=Xcs; fn=fcs; else; Xnew=Xbt; fn=fbt; end

            if fn < fit(i) && rand < A_b(i)
                np(i,:) = Xnew;  fit(i) = fn;
                A_b(i)  = alpha_bat * A_b(i);
                r_b(i)  = 0.5*(1 - exp(-gamma_bat*iter));
            end
        end

        [~,si] = sort(fit,'descend');
        for k = 1:round(pa*N_pop)
            np(si(k),:) = lb + rand(1,n_var).*(ub-lb);
            fit(si(k))  = cbes_fitness(np(si(k),:),t,dt,Ad,Bd,dPL1_o,dPL2_o,M_gl,...
                                       welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,...
                                       seq_len,n_inputs);
        end
        pop = np;
        [cb,ci] = min(fit);
        if cb < bf; bf=cb; Xb=pop(ci,:); end
        cv(iter) = bf;
    end
    conv_all(run,:) = cv;
    fprintf('   Run %d/%d  |  ITAE = %.5f\n', run, N_runs, bf);
    if bf < best_global; best_global=bf; best_params_global=Xb; end
end

p_tfoid_opt = best_params_global(1:6);
p_anfis_opt = best_params_global(7:45);
aw_raw = abs(best_params_global(46));
bw_raw = abs(best_params_global(47));
gw_raw = abs(best_params_global(48));
sw     = aw_raw + bw_raw + gw_raw + 1e-10;
alpha_w_opt = aw_raw/sw;
beta_w_opt  = bw_raw/sw;
gamma_w_opt = gw_raw/sw;

fprintf('\n--- Optimal TFOID Parameters ---\n');
fprintf('  Kp=%.4f  Ki=%.4f  lambda=%.4f\n', p_tfoid_opt(1),p_tfoid_opt(2),p_tfoid_opt(3));
fprintf('  Kd=%.4f  mu=%.4f  sigma=%.4f\n',  p_tfoid_opt(4),p_tfoid_opt(5),p_tfoid_opt(6));
fprintf('--- Layer Mixing Weights ---\n');
fprintf('  alpha(TFOID)=%.3f  beta(ANFIS)=%.3f  gamma(DL)=%.3f\n',...
        alpha_w_opt, beta_w_opt, gamma_w_opt);
fprintf('  Best ITAE = %.5f\n', best_global);

p_pi_fixed = [1.2, 0.8];


fprintf('\n>>> Step 5/5 : Running test cases & generating figures...\n');

%% CASE 1 — Step load disturbance
dPL1_c1 = zeros(Nt,1);  dPL1_c1(t>=1) = dPL_step;
dPL2_c1 = zeros(Nt,1);

[df1_pi, df2_pi, ~, u_pi,  ~      ] = run_simulation(t,dt,Ad,Bd,'PI',    p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c1,dPL2_c1,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
[df1_tf, ~,      ~, u_tf,  ~      ] = run_simulation(t,dt,Ad,Bd,'TFOID', p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c1,dPL2_c1,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
[df1_hy, df2_hy, ~, u_hy, u_dl_c1] = run_simulation(t,dt,Ad,Bd,'HYBRID',p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c1,dPL2_c1,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);

%% CASE 2 — Renewable intermittency
dPL1_c2 = stoch(t,noise_amp,1);  dPL2_c2 = stoch(t,noise_amp*0.8,2);
[df1_pi_r,~,~,~,~] = run_simulation(t,dt,Ad,Bd,'PI',    p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c2,dPL2_c2,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
[df1_tf_r,~,~,~,~] = run_simulation(t,dt,Ad,Bd,'TFOID', p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c2,dPL2_c2,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
[df1_hy_r,~,~,~,~] = run_simulation(t,dt,Ad,Bd,'HYBRID',p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c2,dPL2_c2,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);

%% CASE 3 — Large disturbance + BESS
dPL1_c3 = zeros(Nt,1);  dPL1_c3(t>=1) = dPL_large;
[df1_hy_L,~,~,u_hy_L,~] = run_simulation(t,dt,Ad,Bd,'HYBRID',p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c3,dPL2_c1,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
u_diesel = 0.7*u_hy_L(:,1);
u_bess   = 0.3*u_hy_L(:,1);

%% CASE 4 — Multi-area uncertainty (±20%)
Ad_unc = Ad;  Ad_unc(1,1) = Ad_unc(1,1)*0.8;
[df1_pi4,df2_pi4,~,~,~] = run_simulation(t,dt,Ad,    Bd,'PI',    p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c1,dPL2_c1,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
[df1_hy4,df2_hy4,~,~,~] = run_simulation(t,dt,Ad_unc,Bd,'HYBRID',p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dPL1_c1,dPL2_c1,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);

%% CASE 5 
ITAE_pi = trapz(t, t.*abs(df1_pi));
ITAE_tf = trapz(t, t.*abs(df1_tf));
ITAE_hy = trapz(t, t.*abs(df1_hy));

%% CASE 6 
n_mc = 30;  df_mc = zeros(Nt,n_mc);
for mc = 1:n_mc
    dmc = stoch(t,noise_amp,mc+20);
    [df_mc(:,mc),~,~,~,~] = run_simulation(t,dt,Ad,Bd,'HYBRID',p_pi_fixed,p_tfoid_opt,p_anfis_opt,alpha_w_opt,beta_w_opt,gamma_w_opt,dmc,dmc*0.7,M_gl,welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
end
df_mean = mean(df_mc,2);  df_std = std(df_mc,0,2);

%  PART 14 : PERFORMANCE METRICS

[OS_pi,Ts_pi,Tr_pi,MD_pi] = pmetrics(t,df1_pi);
[OS_tf,Ts_tf,Tr_tf,MD_tf] = pmetrics(t,df1_tf);
[OS_hy,Ts_hy,Tr_hy,MD_hy] = pmetrics(t,df1_hy);

fprintf('\n=======================================================\n');
fprintf('  PERFORMANCE SUMMARY\n');
fprintf('  %-16s %8s %10s %10s %8s\n','Controller','ITAE','Oversh(Hz)','Settle(s)','Rise(s)');
fprintf('  %-16s %8.3f %10.5f %10.2f %8.3f\n','PI',        ITAE_pi,OS_pi,Ts_pi,Tr_pi);
fprintf('  %-16s %8.3f %10.5f %10.2f %8.3f\n','TFOID',     ITAE_tf,OS_tf,Ts_tf,Tr_tf);
fprintf('  %-16s %8.3f %10.5f %10.2f %8.3f\n','Hybrid LFC',ITAE_hy,OS_hy,Ts_hy,Tr_hy);
fprintf('=======================================================\n');

%% =========================================================================
%  PART 15 : ALL FIGURES  — Real results only, no fabricated values
%% =========================================================================

c_pi=[0.85,0.20,0.20]; c_tf=[0.10,0.60,0.20]; c_hy=[0.15,0.35,0.85];
lw=2; fs=12;

%% Fig  DNN Training Loss  (real training output)
figure('Name','Fig1 DNN Training Loss','Color','w','Position',[30,30,650,380]);
valid_loss = dnn_loss(dnn_loss>0);
semilogy(valid_loss,'b-','LineWidth',lw);
xlabel('Epoch','FontSize',fs); ylabel('MSE Loss','FontSize',fs);
title('DNN Training Loss Curve (Adam Optimiser)','FontSize',fs); grid on;

%% Fig  MIANFIS Pre-Training Loss  (new — Section 3.6)
figure('Name','Fig_ANFIS_Pretrain_Loss','Color','w','Position',[30,80,650,380]);
semilogy(anfis_pretrain_loss,'r-','LineWidth',lw);
xlabel('Epoch','FontSize',fs); ylabel('MSE Loss','FontSize',fs);
title('MIANFIS Pre-Training Loss (LSE + Gradient Descent, Section 3.6)','FontSize',fs);
grid on;
text(length(anfis_pretrain_loss), anfis_pretrain_loss(end), ...
     sprintf('  Final MSE = %.5f', anfis_pretrain_loss(end)), ...
     'FontSize',fs-1,'Color',[0.8 0.0 0.0],'FontWeight','bold');

%% Fig  CBES Convergence  (real CBES curve ONLY — PSO/GA not simulated)
figure('Name','Fig3 CBES Convergence','Color','w','Position',[30,130,700,400]);
mean_conv = mean(conv_all,1);
plot(1:T_max, mean_conv, '-o','Color',c_hy,'LineWidth',lw+0.5,...
    'MarkerFaceColor',c_hy,'MarkerSize',5);
xlabel('Iteration','FontSize',fs);
ylabel('Best ITAE','FontSize',fs);
title('CBES Optimisation Convergence','FontSize',fs);
legend('CBES (Proposed)','Location','best','FontSize',fs);
grid on;
% Annotate final ITAE
text(T_max, mean_conv(end), sprintf('  Final ITAE = %.4f', mean_conv(end)),...
    'FontSize',fs-1,'Color',c_hy,'FontWeight','bold');

%% Fig Transient Frequency Response, Area 1 — Step disturbance
figure('Name','Fig4 Step Response Area1','Color','w','Position',[30,180,750,420]);
plot(t,df1_pi,'--','Color',c_pi,'LineWidth',lw,...
    'DisplayName',sprintf('PI     ITAE=%.3f',ITAE_pi)); hold on;
plot(t,df1_tf,'-.','Color',c_tf,'LineWidth',lw,...
    'DisplayName',sprintf('TFOID  ITAE=%.3f',ITAE_tf));
plot(t,df1_hy,'-' ,'Color',c_hy,'LineWidth',lw+0.5,...
    'DisplayName',sprintf('Hybrid ITAE=%.3f',ITAE_hy));
yline(0,'k:','LineWidth',1);
legend('FontSize',fs,'Location','best');
xlabel('Time (s)','FontSize',fs); ylabel('\Deltaf_1 (Hz)','FontSize',fs);
title('Transient Frequency Response — Area 1 (Step Disturbance)','FontSize',fs);
grid on; xlim([0,Tsim]);

%% Fig  Renewable Intermittency Response
figure('Name','Fig6 Renewable Intermittency','Color','w','Position',[30,280,750,420]);
plot(t,df1_pi_r,'--','Color',c_pi,'LineWidth',lw,'DisplayName','PI'); hold on;
plot(t,df1_tf_r,'-.','Color',c_tf,'LineWidth',lw,'DisplayName','TFOID');
plot(t,df1_hy_r,'-' ,'Color',c_hy,'LineWidth',lw+0.5,'DisplayName','Hybrid LFC');
yline(0,'k:','LineWidth',1);
legend('FontSize',fs,'Location','best');
xlabel('Time (s)','FontSize',fs); ylabel('\Deltaf_1 (Hz)','FontSize',fs);
title('Frequency Response under Renewable Intermittency','FontSize',fs);
grid on; xlim([0,Tsim]);

%% Fig  Large Disturbance: Diesel–BESS coordinated dispatch
figure('Name','Fig7 Energy Dispatch','Color','w','Position',[30,30,750,500]);
subplot(2,1,1);
area(t,[u_diesel, u_bess],'FaceAlpha',0.75);
colormap(gca,[0.70 0.40 0.20; 0.20 0.60 0.90]);
legend('Diesel','BESS','Location','best','FontSize',fs);
xlabel('Time (s)','FontSize',fs); ylabel('Power (p.u.)','FontSize',fs);
title('Coordinated Diesel–BESS Dispatch (Large Disturbance)','FontSize',fs);
grid on; xlim([0,Tsim]);
subplot(2,1,2);
plot(t,df1_hy_L,'-','Color',c_hy,'LineWidth',lw); yline(0,'k:','LineWidth',1);
xlabel('Time (s)','FontSize',fs); ylabel('\Deltaf (Hz)','FontSize',fs);
title('Hybrid LFC Frequency Response (Large Disturbance)','FontSize',fs);
grid on; xlim([0,Tsim]);

%% Fig  ITAE Bar Chart  (real computed values)
figure('Name','Fig9 ITAE Comparison','Color','w','Position',[30,430,500,380]);
bh = bar([ITAE_pi, ITAE_tf, ITAE_hy], 0.55, 'FaceColor','flat');
bh.CData = [c_pi; c_tf; c_hy];
set(gca,'XTickLabel',{'PI','TFOID','Hybrid LFC'},'FontSize',fs);
ylabel('ITAE','FontSize',fs);
title('ITAE Comparison — PI vs TFOID vs Hybrid LFC','FontSize',fs);
grid on;
vals = [ITAE_pi, ITAE_tf, ITAE_hy];
for i = 1:3
    text(i, vals(i)+max(vals)*0.02, sprintf('%.4f',vals(i)),...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',fs);
end


%% Fig  WELM-DNN Feedforward Compensation Signal
figure('Name','Fig12 DL Feedforward','Color','w','Position',[30,580,750,380]);
plot(t, u_dl_c1,'-','Color',[0.5,0.0,0.8],'LineWidth',lw);
yline(0,'k:','LineWidth',1);
xlabel('Time (s)','FontSize',fs);
ylabel('u_{DL} (p.u.)','FontSize',fs);
title('WELM-DNN Feedforward Compensation Signal (Layer 3)','FontSize',fs);
grid on; xlim([0,Tsim]);

%% Fig Performance Metrics Bar Panel
figure('Name','Fig14 Performance Metrics','Color','w','Position',[30,30,1050,400]);
metric_names = {'Overshoot (Hz)', 'Max Deviation (Hz)', 'Settling Time (s)'};
vpi = [OS_pi, MD_pi, Ts_pi];
vtf = [OS_tf, MD_tf, Ts_tf];
vhy = [OS_hy, MD_hy, Ts_hy];

for m = 1:3
    subplot(1,3,m);
    bv = [vpi(m), vtf(m), vhy(m)];
    bm = bar(bv, 0.55, 'FaceColor','flat');
    bm.CData = [c_pi; c_tf; c_hy];
    set(gca, 'XTickLabel', {'PI','TFOID','Hybrid'}, 'FontSize', 11);
    title(metric_names{m}, 'FontSize', 11);
    grid on;
    for k = 1:3
        text(k, bv(k) + max(abs(bv))*0.03, sprintf('%.4f', bv(k)), ...
            'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
    end
end
sgtitle('Performance Metrics — PI vs TFOID vs Hybrid LFC', 'FontSize', fs);

%% Fig  FFT of Frequency Deviations (real signals)
figure('Name','Fig17 FFT','Color','w','Position',[30,30,750,420]);
Nfft   = Nt;
f_axis = (0:Nfft-1)*(1/(Nfft*dt));
half   = 1:floor(Nfft/2);
f_axis = f_axis(half);
fft_pi = abs(fft(df1_pi)); fft_pi = fft_pi(half);
fft_tf = abs(fft(df1_tf)); fft_tf = fft_tf(half);
fft_hy = abs(fft(df1_hy)); fft_hy = fft_hy(half);
plot(f_axis,fft_pi,'--','Color',c_pi,'LineWidth',lw,'DisplayName','PI'); hold on;
plot(f_axis,fft_tf,'-.','Color',c_tf,'LineWidth',lw,'DisplayName','TFOID');
plot(f_axis,fft_hy,'-' ,'Color',c_hy,'LineWidth',lw+0.5,'DisplayName','Hybrid LFC');
xlim([0,2]);
legend('FontSize',fs,'Location','best');
xlabel('Frequency (Hz)','FontSize',fs);
ylabel('Magnitude','FontSize',fs);
title('FFT of Frequency Deviations — PI vs TFOID vs Hybrid','FontSize',fs);
grid on;

%% Print performance table to console
fprintf('\n=======================================================\n');
fprintf('  PERFORMANCE SUMMARY (all values from simulation)\n');
fprintf('  %-16s %8s %12s %12s %10s\n',...
    'Controller','ITAE','Oversh(Hz)','Settle(s)','Rise(s)');
fprintf('  %-16s %8.4f %12.5f %12.3f %10.4f\n','PI',       ITAE_pi,OS_pi,Ts_pi,Tr_pi);
fprintf('  %-16s %8.4f %12.5f %12.3f %10.4f\n','TFOID',    ITAE_tf,OS_tf,Ts_tf,Tr_tf);
fprintf('  %-16s %8.4f %12.5f %12.3f %10.4f\n','Hybrid LFC',ITAE_hy,OS_hy,Ts_hy,Tr_hy);
fprintf('  WELM Test MSE: %.6f  |  DNN Test MSE: %.6f\n',welm_mse,dnn_mse);
fprintf('=======================================================\n');
















