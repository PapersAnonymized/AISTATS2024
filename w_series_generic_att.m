%% Clear everything 
clearvars -global;
clear all; close all; clc;

addpath('~/LTTSA/');
addpath('~/ANNLib/');

%% Mem cleanup
%ngpu = gpuDeviceCount();
%for i=1:ngpu
%    reset(gpuDevice(i));
%end

%% Load the data, initialize partition pareameters
%saveDataPrefix = 'wse_';
saveDataPrefix = 'nasdaq0704_';
%saveDataPrefix = 'dj0704_';
%saveDataPrefix = 'nikkei0704_';
%saveDataPrefix = 'dax0704_';

%saveDataPrefix = '7203toyota_';
%saveDataPrefix = 'nvidia_';
%saveDataPrefix = 'tsla4030_';

%saveDataPrefix = 'AirPassengers1_114_30_';
%saveDataPrefix = 'sun_1_';
%saveDataPrefix = 'SN_y_tot_V2.0_spots_4030_';

save_regNet_fileT = '~/data/ws_';

%dataFile = 'wse_data.csv';
dataFile = 'nasdaq_1_3_05-1_28_22.csv';
%dataFile = 'dj_1_3_05-1_28_22.csv';
%dataFile = 'nikkei_1_4_05_1_31_22.csv';
%dataFile = 'dax_1_3_05_1_31_22.csv';

%dataFile = '7203toyota_1_4_05_1_31_22';
%dataFile = 'nvidia_1_3_05_1_28_22';
%dataFile = 'tsla_6_30_10_1_28_22.csv';

%dataFile = 'AirPassengers1.csv';
%dataFile = 'sun_1.csv';
%dataFile = 'SN_y_tot_V2.0_spots.csv';

dataDir = '~/data/STOCKS';
dataFullName = strcat(dataDir,'/',dataFile);

M_off = 1;
%Dilation
M_div = 1;

Me = readmatrix(dataFullName);
[l_whole_ex, ~] = size(Me);

min(Me)
max(Me)
mean(Me)
std(Me)


% input dimesion (days)
m_in = 30;

% input dimesion (parms x days)
x_off = 0;
x_in = 1;
t_in = m_in;

% Try different output dimensions (days)
n_out = 30;

y_off = 0;
y_out = 1;
t_out = n_out;

% Or no future
M = Me;
% Leave space for last full label
l_whole = l_whole_ex - n_out;

% Break the whole dataset in training sessions,
% Set training session length (with m_in datapoints of length m_in), 
l_sess = 6*m_in + n_out;

% Only for 1 whole session (otherwise, comment out)
%l_sess = l_whole;

% No training sessioins that fit into length left after we set aside label
n_sess = floor(l_whole/l_sess);


% Normalization flag
%norm_fl = 1;

ini_rate = 0.01; 
max_epoch = 1000; %250
    

%% regNet parameters
% Fit ann into minimal loss function (SSE)
%mult = 1;
%k_hid1 = floor(mult * (m_in + 1));
%k_hid2 = floor(mult * (2*m_in + 1));

%mb_size = 32;

regNets = cell([n_sess, 1]);
identNets = cell([n_sess, 1]);
       

%% Train or pre-load regNets
for i = 1:n_sess

    norm_fli = 1;
    norm_flo = 1;

    %-regNet = GruNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = GruValNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = LstmValNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    
    %regNet = TransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = VTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = DpTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = DpBatchTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    regNet = Dp2BatchTransNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = RbfNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = TanhNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = SigNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    %regNet = ReluNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);
    %regNet = KgNet2D(x_off, x_in, t_in, y_off, y_out, t_out, ini_rate, max_epoch);

    [regNet, X, Y, Bi, Bo, XI, C, Sx, Sy, k_ob] = regNet.TrainTensors(M, l_sess, n_sess, norm_fli, norm_flo);
    
    modelName = regNet.name;

    save_regNet_file = strcat(save_regNet_fileT, modelName, '_', saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(norm_fli+norm_flo), '_', int2str(n_sess), '.mat');
    %% cross-training
    %%save_regNet_file = strcat(save_regNet_fileT, modelName, '_', saveDataPrefix, int2str(i), '_', int2str(m_in), '_', int2str(n_out), '_', int2str(norm_fl), '_35', '.mat');
    if isfile(save_regNet_file)
        fprintf('Loading net %d from %s\n', i, save_regNet_file);
        load(save_regNet_file, 'regNet');
    else

        %[regNet, X, Y, B, k_ob] = regNet.TrainTensors(M, m_in, n_out, l_sess, n_sess, norm_fl);

        % GPU on
        gpuDevice(1);
        reset(gpuDevice(1));
        
        regNet = regNet.Train(i, X, Y);
        
        % GPU off
        delete(gcp('nocreate'));
        gpuDevice([]);    

        fprintf('Saving %s %d\n', save_regNet_file, i);
        
        save(save_regNet_file, 'regNet');
    end

    regNets{i} = regNet;

end
clear('regNet');


%% Attention Input Identity net
% Train or pre-load Identity nets

useIdentNets = 1;
max_epoch = 250;

for i = 1:n_sess

    identNet = ReluNet2Cl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch);
    %identNet = TanhNet2Cl(x_off, x_in, t_in, n_sess, ini_rate, max_epoch);
    identNet.mb_size = 8*i;%32;
    identNet = identNet.Create();

    dataIdentFile = strcat(save_regNet_fileT, saveDataPrefix, '.ident.', string(i), '.', string(n_sess),...
        '.', string(M_off), '.', string(M_div), '.', dataFile,...
        '.', string(x_off), '.', string(x_in), '.', string(t_in),...
        '.', string(y_off), '.', string(y_out), '.', string(t_out),...
        '.', string(norm_fli), '.', string(norm_flo), '.', string(ini_rate), '.', string(max_epoch), '.mat');

    if useIdentNets ~= 0
        if isfile(dataIdentFile)
            fprintf('Loading Ident net %d from %s\n', i, dataIdentFile);
            load(dataIdentFile, 'identNet');
        else
        
            fprintf('Training Ident net %d\n', i);

            % GPU on
            gpuDevice(1);
            reset(gpuDevice(1));

            tNet = trainNetwork(XI(:, 1:k_ob*i)', C(1:k_ob*i), identNet.lGraph, identNet.options);

            % GPU off
            delete(gcp('nocreate'));
            gpuDevice([]);  

            identNet.trainedNet = tNet;
            identNet.lGraph = tNet.layerGraph; 

            save(dataIdentFile, 'identNet');
        
        end

        identNets{i} = identNet;
    end

end

%% Test parameters 
% the test input period - same as training period, to cover whole data
l_test = l_sess;

% Test from particular training session
sess_off = 0;
% additional offset after training sessions (usually for the future forecast)
offset = 0;

% Left display margin
l_marg = 1;

%% Test parameters for one last session

% Left display margin
%l_marg = 4100;

% Future session
%M = zeros([l_whole_ex+n_out, 1]);
%M(1:l_whole_ex) = Me;
%[l_whole, ~] = size(M);

% Last current session
%l_whole = l_whole_ex;

% Fit last testing session at the end of data
%offset = l_whole - n_sess*l_sess - m_in - n_out;

% Test from particular training session
%sess_off = n_sess-1;


%% For whole-through test, comment out secion above
% Number of training sessions with following full-size test sessions 
t_sess = floor((l_whole - l_test - m_in) / l_sess);

k_tob = 0;
%[X2, Y2, Yh2, Bt, k_tob] = regNets{1}.TestTensors(M, m_in, n_out, l_sess, l_test, t_sess, sess_off, offset, norm_fl);
[X2, Y2, Yh2, Yhs2, Bti, Bto, XI2, Sx2, Sy2, k_tob] = regNets{1}.TestTensors(M, l_sess, l_test, t_sess, sess_off, offset, norm_fli, norm_flo, Bi, Bo, k_tob);

%% test

% GPU on
gpuDevice(1);
reset(gpuDevice(1));

%[X2, Y2] = regNets{1}.Predict(X2, Y2, regNets, t_sess, sess_off, k_tob);
[X2, Y2] = regNets{1}.Predict(X2, Y2, regNets, XI2, identNets, t_sess, sess_off, k_tob);
%[X2, Y2] = regNets{1}.Predict(X2, Y2, regNets, identNets, t_sess, sess_off, k_tob);

% GPU off
delete(gcp('nocreate'));
gpuDevice([]);

%% re-scale in observation bounds
%%if(norm_fl)
%    [Y2, Yh2] = regNets{1}.ReScale(Y2, Yh2, Bt, t_sess, sess_off, k_tob);
%%end

%if(norm_fli)
%    [X, X2] = regNets{1}.ReScaleIn(X, X2, Bi, n_sess, t_sess, sess_off, k_ob, k_tob);
%end

if(norm_flo)
    [Y, Y2, Yhs2] = regNets{1}.ReScaleOut(Y, Y2, Yhs2, Bo, Bto, n_sess, t_sess, sess_off, k_ob, k_tob);
end

%% Calculate errors
%[S2, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = regNets{1}.Calc_mape(Y2, Yh2, n_out); 

%fprintf('%s, trainN %s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, MAPErr: %f+-%f MaxAPErr %f+-%f\n', modelName, saveDataPrefix, dataFile, norm_fl, m_in, n_out, n_sess, t_sess, S2, S2Std, mean(ma_err), std(ma_err));


%[S2Q, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = regNets{1}.Calc_rmse(Y2, Yh2, n_out); 

%fprintf('%s, trainN %s, dataFN %s, NormF:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, RMSErr: %f+-%f MaxRSErr %f+-%f\n', modelName, saveDataPrefix, dataFile, norm_fl, m_in, n_out, n_sess, t_sess, S2Q, S2StdQ, mean(ma_errQ), std(ma_errQ));

[Em, S2, S2Mean, S2Std, S2s, ma_err, sess_ma_idx, ob_ma_idx, mi_err, sess_mi_idx, ob_mi_idx] = regNets{1}.Calc_mape(Y2, Yh2); 

fprintf('%s, dataFN %s, NormFi:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, MAPErr: %f+-%f MeanMaxAPErr %f+-%f\n', modelName, dataFile, norm_fli, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2, S2Std, mean(ma_err), std(ma_err));


[Er, S2Q, S2MeanQ, S2StdQ, S2sQ, ma_errQ, sess_ma_idxQ, ob_ma_idxQ, mi_errQ, sess_mi_idxQ, ob_mi_idxQ] = regNets{1}.Calc_rmse(Y2, Yh2); 

fprintf('%s, dataFN %s, NormFi:%d, M_in:%d, N_out:%d, Tr_sess:%d, Ts_sess:%d, RMSErr: %f+-%f MeanMaxRSErr %f+-%f\n', modelName, dataFile, norm_fli, regNets{1}.m_in, regNets{1}.n_out, n_sess, t_sess, S2Q, S2StdQ, mean(ma_errQ), std(ma_errQ));
%%
% Write per-session errors to a file
fd = fopen( strcat('ws_att_err_', saveDataPrefix, dataFile, '.', regNets{1}.name, '.', string(useIdentNets), '.txt'),'w' );

fprintf(fd, "Sess MeanPE MeanRSE\n");

for i = 1:t_sess-sess_off
    fprintf(fd, "%d %f %f\n", i, S2s(i), S2sQ(i));
end
fclose(fd);
%% Error and Series Plot
yLab = strcat("Index ", saveDataPrefix);
%regNets{1}.Err_graph(M, l_whole_ex, Y2, l_whole, l_sess, m_in, n_out, k_tob, t_sess, sess_off, offset, l_marg);
regNets{1}.Err_graph(M, Em, Er, l_whole_ex, Y2, Sy2, l_whole, l_sess, k_tob, t_sess, sess_off, offset, l_marg, modelName, yLab);
