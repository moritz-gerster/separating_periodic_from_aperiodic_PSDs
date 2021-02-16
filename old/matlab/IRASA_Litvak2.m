
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MATLAB script for the extraction of rhythmic spectral features
% from the electrophysiological signal based on Irregular Resampling
% Auto-Spectral Analysis (IRASA, Wen & Liu, Brain Topogr. 2016)
%
% Ensure FieldTrip is correcty added to the MATLAB path:
%   addpath <path to fieldtrip home directory>
%   ft_defaults
%
% From Stolk et al., Electrocorticographic dissociation of alpha and
% beta rhythmic activity in the human sensorimotor system

% script from https://www.fieldtriptoolbox.org/example/irasa/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
% download human ECoG dataset from https://osf.io/z4hfm/
disp(['hhhhhh'])
%load('S5_raw_segmented.mat')
load('../../data/raw/rest/subj2/on/subj2_on_R5.mat')

% filter %% and re-reference the raw data
cfg               = [];

% segment the data into one-second non-overlapping chunks
cfg               = [];
cfg.length        = 1;
cfg.overlap       = 0;
data_c = ft_redefinetrial(cfg, data);

% partition the data into ten overlapping sub-segments
w = data.time{1}(end)-data.time{1}(1); % window length
cfg               = [];
cfg.length        = w*.9;
cfg.overlap       = 1-(((w-cfg.length)/cfg.length)/(10-1));

% perform IRASA and regular spectral analysis
cfg               = [];
cfg.foilim        = [1 100]; % freq range
cfg.taper         = 'hanning';
cfg.pad           = 'nextpow2';
cfg.method        = 'irasa';
frac_r = ft_freqanalysis(cfg, data);
cfg.method        = 'mtmfft';
orig_r = ft_freqanalysis(cfg, data);

frac = frac_r
orig = orig_r
% sensorimotor channels
crtx_stn = {'SMA','leftM1','rightM1','STN_L01','STN_L12', ...
  'STN_L23','STN_R01','STN_R12','STN_R23'};


% subtract the fractal component from the power spectrum
cfg               = [];
cfg.parameter     = 'powspctrm';
cfg.operation     = 'x2-x1';
osci = ft_math(cfg, frac, orig);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%osci.powspctrm(osci.powspctrm < 0) = 0;%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%% no gauss fitting here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% plot the fractal component and the power spectrum 
figure; plot(frac.freq, mean(frac.powspctrm), ...
  'linewidth', 3, 'color', [0 0 0])
hold on; plot(orig.freq, mean(orig.powspctrm), ...
  'linewidth', 3, 'color', [.6 .6 .6])
hold on; plot(osci.freq, mean(osci.powspctrm), ...
      'linewidth', 3, 'color', [.3 .3 .3])

set(gca, 'YScale', 'log')
legend('Fractal component', 'Power spectrum', 'Oscillatory Component');
xlabel('Frequency'); ylabel('Power');

%saveas(gcf, '../../plots/Irasa_all_chans.pdf')

%% plot the fractal component and the power spectrum 
figure; plot(frac.freq, mean(frac.powspctrm), ...
  'linewidth', 3, 'color', [0 0 0])

set(gca, 'YScale', 'log')
set(gca, 'YLim', [0 1])
set(gca, 'XScale', 'log')
legend('Fractal component');
xlabel('Frequency'); ylabel('Power');

%%
figure; plot(osci.freq, mean(osci.powspctrm), ...
      'linewidth', 3, 'color', [.3 .3 .3])
set(gca, 'YLim', [0 1])
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')
legend('Oscillatory Component');
xlabel('Frequency'); ylabel('Power');

%% plot the fractal component and the power spectrum 
figure; plot(orig.freq, mean(orig.powspctrm), ...
  'linewidth', 3, 'color', [0 0 0])

set(gca, 'YScale', 'log')
set(gca, 'YLim', [0 1])
set(gca, 'XScale', 'log')
legend('Power spectrum');
xlabel('Frequency'); ylabel('Power');