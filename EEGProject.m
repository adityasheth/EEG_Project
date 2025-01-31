clear; clc;

% Assigning train data directory & cahnging working directory
data_dir = 'E:/Anger EEG/EEG_project';
cd(data_dir)

loadFile = fullfile(data_dir, 'processedData_20151209_1514__0_1_50_hz.mat');
load(loadFile);

% Defining variable(features)
DELTA_POWER = zeros(1,31); freqrange_delta = [0 4];
THETA_POWER = zeros(1,31); freqrange_theta = [4 7];
ALPHA_POWER = zeros(1,31); freqrange_alpha = [7 13];
BETA_POWER = zeros(1,31); freqrange_beta = [13 30];
GAMMA_POWER = zeros(1,31); freqrange_gamma = [30 50];

SNR_DELTA_FINAL = zeros(1,32); THD_DELTA_FINAL = zeros(1,32); SINAD_DELTA_FINAL = zeros(1,32);
SNR_THETA_FINAL = zeros(1,32); THD_THETA_FINAL = zeros(1,32); SINAD_THETA_FINAL = zeros(1,32); 
SNR_ALPHA_FINAL = zeros(1,32); THD_ALPHA_FINAL = zeros(1,32); SINAD_ALPHA_FINAL = zeros(1,32);
SNR_BETA_FINAL = zeros(1,32); THD_BETA_FINAL = zeros(1,32); SINAD_BETA_FINAL = zeros(1,32);
SNR_GAMMA_FINAL = zeros(1,32); THD_GAMMA_FINAL = zeros(1,32); SINAD_GAMMA_FINAL = zeros(1,32);

film_start = 0; film_end = 0; epochEndTime = 120; EEG.srate = 256;

for iVideo = 1:3
    film_start = floor(EEG.event(1,iVideo).latency);
    film_end = floor(EEG.event(1,iVideo).latency) + floor(epochEndTime * EEG.srate);
    data = EEG.data(:,film_start:film_end)';
    
    % add code of feat extraction below
      Fs = 500; % sampling_frequency = 256 Hz
    % Ts = 289999; % 289999/256
    % [N,nu]=size(data);%obtain size of data
    
% Five frequency bands
%   delta = 1-4 Hz
%   theta = 4-7 Hz
%   alpha = 7-13 Hz
%   beta = 13-30 Hz
%   gamma = 30-50 Hz

%DELTA - BAND PASS FILTER (0-4)
% Calculating band power
del_pow = bandpower(data, Fs, freqrange_delta); 
DELTA_POWER = cat(1,DELTA_POWER,del_pow);

%THETA- BAND PASS FILTER (4-7)
% Calculating band power
theta_pow = bandpower(data, Fs, freqrange_theta); 
THETA_POWER = cat(1,THETA_POWER,theta_pow);

%ALPHA BAND PASS FILTER (8-12)
% Calculating band power
alpha_pow = bandpower(data, Fs, freqrange_alpha); 
ALPHA_POWER = cat(1,ALPHA_POWER,alpha_pow);

%BETA  BAND PASS FILTER (12-30)
% Calculating band power
beta_pow = bandpower(data, Fs, freqrange_beta); 
BETA_POWER = cat(1,BETA_POWER,beta_pow);

%GAMMA - BAND PASS FILTER (4-7)
% Calculating band power
gamma_pow = bandpower(data, Fs, freqrange_gamma); 
GAMMA_POWER = cat(1,GAMMA_POWER,gamma_pow);

% DELTA BAND
Fs = 500;                 % Sampling Frequency
Fstop1 = 0.5;             % First Stopband Frequency
Fpass1 = 1;               % First Passband Frequency
Fpass2 = 4;               % Second Passband Frequency
Fstop2 = 4.5;             % Second Stopband Frequency
Dstop1 = 0.001;           % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                       0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b2 = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b2);
x1=filter(Hd2,data);

% THETA BAND
Fs = 500;                 % Sampling Frequency
Fstop1 = 3.5;             % First Stopband Frequency
Fpass1 = 4;               % First Passband Frequency
Fpass2 = 7;               % Second Passband Frequency
Fstop2 = 7.5;             % Second Stopband Frequency
Dstop1 = 0.001;           % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                       0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b2 = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b2);
x2=filter(Hd2,data);

% ALPHA BAND
Fs = 500;                 % Sampling Frequency
Fstop1 = 6.5;             % First Stopband Frequency
Fpass1 = 7;               % First Passband Frequency
Fpass2 = 13;               % Second Passband Frequency
Fstop2 = 13.5;             % Second Stopband Frequency
Dstop1 = 0.001;           % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                       0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b2 = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b2);
x3=filter(Hd2,data);

% BETA BAND
Fs = 500;                 % Sampling Frequency
Fstop1 = 12.5;             % First Stopband Frequency
Fpass1 = 13;               % First Passband Frequency
Fpass2 = 30;               % Second Passband Frequency
Fstop2 = 30.5;             % Second Stopband Frequency
Dstop1 = 0.001;           % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                       0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b2 = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b2);
x4=filter(Hd2,data);

% GAMMA BAND
Fs = 500;                 % Sampling Frequency
Fstop1 = 29.5;             % First Stopband Frequency
Fpass1 = 30;               % First Passband Frequency
Fpass2 = 50;               % Second Passband Frequency
Fstop2 = 50.5;             % Second Stopband Frequency
Dstop1 = 0.001;           % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                       0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b2 = firpm(N, Fo, Ao, W, {dens});
Hd2 = dfilt.dffir(b2);
x5=filter(Hd2,data);

x1=x1';
x2=x2';
x3=x3';
x4=x4';
x5=x5';

SNR_DELTA = zeros(size(x1,1),1); THD_DELTA = zeros(size(x1,1),1); SINAD_DELTA = zeros(size(x1,1),1);
SNR_THETA = zeros(size(x2,1),1); THD_THETA = zeros(size(x2,1),1); SINAD_THETA = zeros(size(x2,1),1); 
SNR_ALPHA = zeros(size(x3,1),1); THD_ALPHA = zeros(size(x3,1),1); SINAD_ALPHA = zeros(size(x3,1),1);
SNR_BETA = zeros(size(x4,1),1); THD_BETA = zeros(size(x4,1),1); SINAD_BETA = zeros(size(x4,1),1);
SNR_GAMMA = zeros(size(x5,1),1); THD_GAMMA = zeros(size(x5,1),1); SINAD_GAMMA = zeros(size(x5,1),1);

for k = 1:size(data,2)
    
    % Extracting Electrode-wise Sound to Noise Ratio
    snr_delta = snr(x1(k,:),Fs);
    snr_theta = snr(x2(k,:),Fs);
    snr_alpha = snr(x3(k,:),Fs);
    snr_beta = snr(x4(k,:),Fs);
    snr_gamma = snr(x5(k,:),Fs);
    
    SNR_DELTA(k) = snr_delta;
    SNR_THETA(k) = snr_delta;
    SNR_ALPHA(k) = snr_delta;
    SNR_BETA(k) = snr_delta;
    SNR_GAMMA(k) = snr_delta;
    
    % Extracting electrode-wise Total Harmonic Distribution
    thd_delta = thd(x1(k,:),Fs);
    thd_theta = thd(x2(k,:),Fs);
    thd_alpha = thd(x3(k,:),Fs);
    thd_beta = thd(x4(k,:),Fs);
    thd_gamma = thd(x5(k,:),Fs);
    
    THD_DELTA(k) = thd_delta;
    THD_THETA(k) = thd_delta;
    THD_ALPHA(k) = thd_delta;
    THD_BETA(k) = thd_delta;
    THD_GAMMA(k) = thd_delta;
    
    % Extracting electrode-wise SINAD
    sinad_delta = sinad(x1(k,:),Fs);
    sinad_theta = sinad(x2(k,:),Fs);
    sinad_alpha = sinad(x3(k,:),Fs);
    sinad_beta = sinad(x4(k,:),Fs);
    sinad_gamma = sinad(x5(k,:),Fs);
    
    SINAD_DELTA(k) = sinad_delta;
    SINAD_THETA(k) = sinad_delta;
    SINAD_ALPHA(k) = sinad_delta;
    SINAD_BETA(k) = sinad_delta;
    SINAD_GAMMA(k) = sinad_delta;
    
end

class_label = iVideo*ones(1,1);

SNR_DELTA = horzcat(SNR_DELTA',class_label);
SNR_DELTA_FINAL = [SNR_DELTA_FINAL;SNR_DELTA];
SNR_THETA = horzcat(SNR_THETA',class_label);
SNR_THETA_FINAL = [SNR_THETA_FINAL;SNR_THETA];
SNR_ALPHA = horzcat(SNR_ALPHA',class_label);
SNR_ALPHA_FINAL = [SNR_ALPHA_FINAL;SNR_ALPHA];
SNR_BETA = horzcat(SNR_BETA',class_label);
SNR_BETA_FINAL = [SNR_BETA_FINAL;SNR_BETA];
SNR_GAMMA = horzcat(SNR_GAMMA',class_label);
SNR_GAMMA_FINAL = [SNR_GAMMA_FINAL;SNR_GAMMA];

THD_DELTA = horzcat(THD_DELTA',class_label);
THD_DELTA_FINAL = [THD_DELTA_FINAL;THD_DELTA];
THD_THETA = horzcat(THD_THETA',class_label);
THD_THETA_FINAL = [THD_THETA_FINAL;THD_THETA];
THD_ALPHA = horzcat(THD_ALPHA',class_label);
THD_ALPHA_FINAL = [THD_ALPHA_FINAL;THD_ALPHA];
THD_BETA = horzcat(THD_BETA',class_label);
THD_BETA_FINAL = [THD_BETA_FINAL;THD_BETA];
THD_GAMMA = horzcat(THD_GAMMA',class_label);
THD_GAMMA_FINAL = [THD_GAMMA_FINAL;THD_GAMMA];

SINAD_DELTA = horzcat(SINAD_DELTA',class_label);
SINAD_DELTA_FINAL = [SINAD_DELTA_FINAL;SINAD_DELTA];
SINAD_THETA = horzcat(SINAD_THETA',class_label);
SINAD_THETA_FINAL = [SINAD_THETA_FINAL;SINAD_THETA];
SINAD_ALPHA = horzcat(SINAD_ALPHA',class_label);
SINAD_ALPHA_FINAL = [SINAD_ALPHA_FINAL;SINAD_ALPHA];
SINAD_BETA = horzcat(SINAD_BETA',class_label);
SINAD_BETA_FINAL = [SINAD_BETA_FINAL;SINAD_BETA];
SINAD_GAMMA = horzcat(SINAD_GAMMA',class_label);
SINAD_GAMMA_FINAL = [SINAD_GAMMA_FINAL;SINAD_GAMMA];


end 

DELTA_POWER = DELTA_POWER(2:end,:);
THETA_POWER = THETA_POWER(2:end,:);
ALPHA_POWER = ALPHA_POWER(2:end,:);
BETA_POWER = BETA_POWER(2:end,:);
GAMMA_POWER = GAMMA_POWER(2:end,:);

label = [1 1 1 1 1 2 2 2 2 2 3 3 3 3 3]';

POWER = cat(1, DELTA_POWER, THETA_POWER, ALPHA_POWER, BETA_POWER, GAMMA_POWER);
POWER = horzcat(POWER,label);

SNR_DELTA_FINAL = SNR_DELTA_FINAL(2:end,:);
SNR_THETA_FINAL = SNR_THETA_FINAL(2:end,:);
SNR_ALPHA_FINAL = SNR_ALPHA_FINAL(2:end,:);
SNR_BETA_FINAL = SNR_BETA_FINAL(2:end,:);
SNR_GAMMA_FINAL = SNR_GAMMA_FINAL(2:end,:);

SNR = cat(1,SNR_DELTA_FINAL, SNR_THETA_FINAL, SNR_ALPHA_FINAL, SNR_BETA_FINAL, SNR_GAMMA_FINAL);

THD_DELTA_FINAL = THD_DELTA_FINAL(2:end,:);
THD_THETA_FINAL = THD_THETA_FINAL(2:end,:);
THD_ALPHA_FINAL = THD_ALPHA_FINAL(2:end,:);
THD_BETA_FINAL = THD_BETA_FINAL(2:end,:);
THD_GAMMA_FINAL = THD_GAMMA_FINAL(2:end,:);

THD = cat(1,THD_DELTA_FINAL, THD_THETA_FINAL, THD_ALPHA_FINAL, THD_BETA_FINAL, THD_GAMMA_FINAL);

SINAD_DELTA_FINAL = SINAD_DELTA_FINAL(2:end,:);
SINAD_THETA_FINAL = SINAD_THETA_FINAL(2:end,:);
SINAD_ALPHA_FINAL = SINAD_ALPHA_FINAL(2:end,:);
SINAD_BETA_FINAL = SINAD_BETA_FINAL(2:end,:);
SINAD_GAMMA_FINAL = SINAD_GAMMA_FINAL(2:end,:);

SINAD = cat(1,SINAD_DELTA_FINAL, SINAD_THETA_FINAL, SINAD_ALPHA_FINAL, SINAD_BETA_FINAL, SINAD_GAMMA_FINAL);

FEATURE_VECTOR = [POWER;SNR;THD;SINAD];

save('Feature_Vector','FEATURE_VECTOR')
Feat_vec = load('Feature_vector.mat');
csvwrite('Feature_Vector.csv', Feat_vec.FEATURE_VECTOR);



