clear; clc;

% Assigning train data directory & cahnging working directory
data_dir = 'E:/Anger EEG';
cd(data_dir)

loadFile = fullfile(data_dir, 'processedData_20151209_1514__0_1_50_hz.mat');
load(loadFile);

film_start = 0; film_end = 0; epochEndTime = 120; EEG.srate = 256;
for iVideo = 1:3
    film_start = floor(EEG.event(1,iVideo).latency);
    film_end = floor(EEG.event(1,iVideo).latency) + floor(epochEndTime * EEG.srate);
    EEG_data = EEG.data(:,film_start:film_end)';
    % add code of feat extraction below
    Fs = 240; % sampling_frequency = 240 Hz
     
    N = length(EEG_data);
    Y = fft(EEG_data);
    Y = Y(1:N/2+1);
    psd = (1/(Fs*N)) * abs(Y).^2;
    psd(2:end-1) = 2*psd(2:end-1);
    freq = 0:Fs/length(EEG_data):Fs/2;

    plot(freq,10*log10(psd))
    grid on
    title('Periodogram Using FFT')
    xlabel('Frequency (Hz)')
    ylabel('Power/Frequency (dB/Hz)')
    
% Three frequency bands
%   theta = 4-7 Hz
%   alpha = 8 -13 Hz
%   beta = 13-30 Hz
% Calculate sum of squared absolute values within each frequency band
% Perform Azimuthal transformation of electrode locations;
% Use Clough-Tocher Scheme for interpolating the scattered power elements

% FFT power values extracted for three frequency bands (theta, alpha, beta). 
% Features are arranged in band and electrodes order (theta_1, theta_2..., theta_64, alpha_1, alpha_2, ..., beta_64). 
% There are seven time windows, features for each time window are aggregated sequentially (i.e. 0:191 --> time window 1, 192:383 --> time windw 2 and so on. 
% Last column contains the class labels (load levels).
end   