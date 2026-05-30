%% DNN train / predict 
function [net, loss_hist] = dnn_train(net, X_tr, Y_tr, X_val, Y_val, opts)
    lr=opts.lr; epochs=opts.epochs; bs=opts.batch;
    l2=opts.l2; do_rate=opts.dropout; pat=opts.patience;
    b1=0.9; b2=0.999; eps_a=1e-8;
    n_tr=size(X_tr,1); loss_hist=zeros(epochs,1);
    best_val=inf; best_net=net; wait=0;

    for ep=1:epochs
        idx=randperm(n_tr); X_s=X_tr(idx,:); Y_s=Y_tr(idx,:);
        ep_loss=0; n_batch=floor(n_tr/bs);
        for b=1:n_batch
            xb=X_s((b-1)*bs+1:b*bs,:); yb=Y_s((b-1)*bs+1:b*bs,:);
            A=cell(net.n_layer+1,1); Z=cell(net.n_layer,1); mask=cell(net.n_layer,1);
            A{1}=xb;
            for l=1:net.n_layer
                Z{l}=A{l}*net.W{l}+repmat(net.b{l},bs,1);
                if l<net.n_layer
                    A{l+1}=relu(Z{l});
                    if do_rate>0
                        mask{l}=(rand(size(A{l+1}))>do_rate)/(1-do_rate);
                        A{l+1}=A{l+1}.*mask{l};
                    end
                else; A{l+1}=Z{l}; end
            end
            err=A{end}-yb; ep_loss=ep_loss+mean(err(:).^2);
            dA=cell(net.n_layer+1,1); dA{end}=err/bs;
            for l=net.n_layer:-1:1
                dZ=dA{l+1};
                if l<net.n_layer
                    dZ=dZ.*(Z{l}>0);
                    if do_rate>0&&~isempty(mask{l}); dZ=dZ.*mask{l}; end
                end
                dW=A{l}'*dZ+l2*net.W{l}; db=sum(dZ,1); dA{l}=dZ*net.W{l}';
                net.t_adam=net.t_adam+1; tt=net.t_adam;
                net.mW{l}=b1*net.mW{l}+(1-b1)*dW; net.vW{l}=b2*net.vW{l}+(1-b2)*(dW.^2);
                net.W{l}=net.W{l}-lr*(net.mW{l}/(1-b1^tt))./(sqrt(net.vW{l}/(1-b2^tt))+eps_a);
                net.mb{l}=b1*net.mb{l}+(1-b1)*db; net.vb{l}=b2*net.vb{l}+(1-b2)*(db.^2);
                net.b{l}=net.b{l}-lr*(net.mb{l}/(1-b1^tt))./(sqrt(net.vb{l}/(1-b2^tt))+eps_a);
            end
        end
        loss_hist(ep)=ep_loss/n_batch;
        y_val_pred=dnn_predict(net,X_val); val_loss=mean((y_val_pred-Y_val).^2,'all');
        if val_loss<best_val; best_val=val_loss; best_net=net; wait=0;
        else; wait=wait+1; if wait>=pat
            fprintf('     Early stopping at epoch %d (val_loss=%.5f)\n',ep,val_loss); break; end
        end
    end
    net=best_net;
end