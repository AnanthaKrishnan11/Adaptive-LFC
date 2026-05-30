function itae=cbes_fitness(params_all,t,dt,Ad,Bd,dPL1,dPL2,M_gl,...
    welm_model,dnn_net,X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs)
    try
        p_tf=params_all(1:6); p_an=params_all(7:45);
        aw=abs(params_all(46)); bw=abs(params_all(47)); gw=abs(params_all(48));
        sw=aw+bw+gw+1e-10; aw=aw/sw; bw=bw/sw; gw=gw/sw;
        [df1,df2,~,~,~]=run_simulation(t,dt,Ad,Bd,'HYBRID',[0,0],p_tf,p_an,...
            aw,bw,gw,dPL1,dPL2,M_gl,welm_model,dnn_net,...
            X_mu,X_sig,Y_mu,Y_sig,seq_len,n_inputs);
        itae=trapz(t,t.*(abs(df1)+abs(df2)));
    catch; itae=1e6; end
end