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
    data = EEG.data(:,film_start:film_end)';
    % add code of feat extraction below
    Fs = 256; % sampling_frequency = 256 Hz
    % Ts = 289999; % 289999/256
    [N,nu]=size(data);%obtain size of data
    
    % part-2
    y=fft(data);% fft of data
    ps1=abs(y).^2;% power spectrum using fft
    freq=(1:N)*Fs/N;%frequency vector
    h2=figure
    plot(freq,20*log(ps1),'b')
    title('POWER SPECTRUM USING FFT METHOD')
     
%     N = length(EEG_data);
%     Y = fft(EEG_data);
%     Y = Y(1:N/2+1);
%     psd = (1/(Fs*N)) * abs(Y).^2;
%     psd(2:end-1) = 2*psd(2:end-1);
%     freq = 0:Fs/length(EEG_data):Fs/2;
% 
%     plot(freq,10*log10(psd))
%     grid on
%     title('Periodogram Using FFT')
%     xlabel('Frequency (Hz)')
%     ylabel('Power/Frequency (dB/Hz)')
    
% Three frequency bands
%   theta = 4-7 Hz
%   alpha = 8 -13 Hz
%   beta = 13-30 Hz

%THETA- BAND PASS FILTER (4-7)

Fs = 500;  % Sampling Frequency
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
h11=figure
plot(x2,'r') % plot(t,x2,'r')
title('waveform for THETA band')

%FREQUENCY SPECTRUM OF THETA 
L=10;
Fs=500;
NFFT = 2^nextpow2(L); % Next power of 2 from length of x2
Y2 = fft(x2,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2);
% Plot single-sided amplitude spectrum THETA 
h12=figure
plot(f,2*abs(Y2(1:NFFT/2))) 
title('Single-Sided Amplitude Spectrum of THETA x2(t)')
xlabel('Frequency (Hz)')
ylabel('|Y2(f)|')



%ALPHA BAND PASS FILTER (8-12)

Fs = 500;  % Sampling Frequency
Fstop1 = 7.5;             % First Stopband Frequency
Fpass1 = 8;               % First Passband Frequency
Fpass2 = 12;              % Second Passband Frequency
Fstop2 = 12.5;            % Second Stopband Frequency
Dstop1 = 0.0001;          % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b3  = firpm(N, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
x3=filter(Hd3,data);
h13=figure
plot(x3,'r') % plot(t,x3,'r')
title('waveform for ALPHA band')
%FREQUENCY SPECTRUM OF ALPHA BAND
L=10;
Fs=500;
NFFT = 2^nextpow2(L); % Next power of 2 from length of x3
Y3 = fft(x3,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2);
% Plot single-sided amplitude spectrum ALPHA
h14=figure
plot(f,2*abs(Y3(1:NFFT/2))) 
title('Single-Sided Amplitude Spectrum of ALPHA x3(t)')
xlabel('Frequency (Hz)')
ylabel('|Y3(f)|')


%BETA  BAND PASS FILTER (12-30)

Fs = 500;  % Sampling Frequency

Fstop1 = 11.5;            % First Stopband Frequency
Fpass1 = 12;              % First Passband Frequency
Fpass2 = 30;              % Second Passband Frequency
Fstop2 = 30.5;            % Second Stopband Frequency
Dstop1 = 0.0001;          % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor

% Calculate the order from the parameters using FIRPMORD.
[N, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1 ...
                          0], [Dstop1 Dpass Dstop2]);

% Calculate the coefficients using the FIRPM function
b4   = firpm(N, Fo, Ao, W, {dens});
Hd4 = dfilt.dffir(b4);
x4=filter(Hd4,data);
h15=figure
plot(x4,'r') %plot(t,x4,'r')
title('waveform for BETA band')
%Frequency spectrum of BETA band
L=10;
Fs=500;
NFFT = 2^nextpow2(L); % Next power of 2 from length of x4
Y4 = fft(x4,NFFT)/L;
f = Fs/2*linspace(0,1,NFFT/2);
% Plot single-sided amplitude spectrum BETA
h16=figure
plot(f,2*abs(Y4(1:NFFT/2))) 
title('Single-Sided Amplitude Spectrum of BETA x4(t)')
xlabel('Frequency (Hz)')
ylabel('|Y4(f)|')


% Calculate sum of squared absolute values within each frequency band for
% each electrode
% Perform Azimuthal transformation of electrode locations;
% Use Clough-Tocher Scheme for interpolating the scattered power elements

% FFT power values extracted for three frequency bands (theta, alpha, beta). 
% Features are arranged in band and electrodes order (theta_1, theta_2..., theta_64, alpha_1, alpha_2, ..., beta_64). 
% There are seven time windows, features for each time window are aggregated sequentially (i.e. 0:191 --> time window 1, 192:383 --> time windw 2 and so on. 
% Last column contains the class labels (load levels).
% What I need ? 1) Location of each electrode(3-D Coordinates)
end   