
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

%%%% should be irrelevant because data is high-low pass filtered
cfg.hpfilter      = 'yes'; % high-pass in order to get rid of low-freq trends
cfg.hpfiltord     = 3;
cfg.hpfreq        = 1;
cfg.lpfilter      = 'yes'; % low-pass in order to get rid of high-freq noise
cfg.lpfiltord     = 3;
cfg.lpfreq        = 249; % 249 when combining with a linenoise bandstop filter
cfg.bsfilter      = 'yes'; % band-stop filter, to take out 50 Hz and its harmonics
cfg.bsfiltord     = 3;
cfg.bsfreq        = [49 51; 99 101; 149 151; 199 201]; % EU line noise
cfg.reref         = 'yes'; %%%% why rereference?
cfg.refchannel    = 'all';
data_f = ft_preprocessing(cfg, data);

% segment the data into one-second non-overlapping chunks
cfg               = [];
cfg.length        = 1;
cfg.overlap       = 0;
data_c = ft_redefinetrial(cfg, data_f);

% partition the data into ten overlapping sub-segments
w = data_c.time{1}(end)-data_c.time{1}(1); % window length
cfg               = [];
cfg.length        = w*.9;
cfg.overlap       = 1-(((w-cfg.length)/cfg.length)/(10-1));
data_r = ft_redefinetrial(cfg, data_c);

% perform IRASA and regular spectral analysis
cfg               = [];
cfg.foilim        = [1 100]; % freq range
cfg.taper         = 'hanning';
cfg.pad           = 'nextpow2';
cfg.keeptrials    = 'yes'; %??
cfg.method        = 'irasa';
frac_r = ft_freqanalysis(cfg, data_r);
cfg.method        = 'mtmfft';
orig_r = ft_freqanalysis(cfg, data_r);

% average across the sub-segments
frac_s = {}; 
orig_s = {};
for rpt = unique(frac_r.trialinfo(:,end))'
  cfg               = [];
  cfg.trials        = find(frac_r.trialinfo(:,end)==rpt);
  cfg.avgoverrpt    = 'yes';
  frac_s{end+1} = ft_selectdata(cfg, frac_r);
  orig_s{end+1} = ft_selectdata(cfg, orig_r);
end
frac_a = ft_appendfreq([], frac_s{:});
orig_a = ft_appendfreq([], orig_s{:});


% sensorimotor channels
crtx_stn = {'SMA','leftM1','rightM1','STN_L01','STN_L12', ...
  'STN_L23','STN_R01','STN_R12','STN_R23'};

% average across trials
cfg               = [];
cfg.trials        = 'all';
cfg.avgoverrpt    = 'yes';
cfg.channel       = crtx_stn;
frac = ft_selectdata(cfg, frac_a);
orig = ft_selectdata(cfg, orig_a);

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


