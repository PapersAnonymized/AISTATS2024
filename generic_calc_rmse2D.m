function [E2f, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx]=generic_calc_rmse2D(Y2, Yh2, n_out)
  
    %E2f(:, :, :) = ((Y2(1:n_out, :, :) - Yh2(1:n_out, :, :)) ./ Yh2(1:n_out, :, :)).^2;
    E2f(:, :, :) = ((Y2(1:n_out, :, :) - Yh2(1:n_out, :, :))).^2;
    [skn, sjf, sif] = size(E2f);

    S2 = sum(E2f, [2,3]);
    Sn2 = sjf*sif;
    
    S2 = sum(sqrt(S2/Sn2), 1)/skn;

    [m, i] = max(E2f);
    [ma_err, sess_ma_idx]=max(m);
    ob_ma_idx = i(sess_ma_idx);

    [m, i] = min(E2f);
    [mi_err, sess_mi_idx]=min(m);
    ob_mi_idx = i(sess_mi_idx);

    % Per session mean+std
    S2s = sum(E2f, 2);
    Sn2s = sjf;
    
    S2s = sum(sqrt(S2s/Sn2s), 1)/skn;
    S2Mean = mean(S2s);
    S2Std = std(S2s);
end