function y_pred = welm_predict(model, X)
    H      = relu(X * model.Win + repmat(model.b_in, size(X,1), 1));
    y_pred = H * model.beta;   % [n_samples × n_outputs]
end