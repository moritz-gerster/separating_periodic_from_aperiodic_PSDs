freqs = 2.^([0:0.5:2,2.3:0.3:5,5.2:0.2:8])

data_on = load('/Users/moritzgerster/Documents/Code/Litvak11/data/raw/subj1/on/R1.mat', 'data')
data_off = load('/Users/moritzgerster/Documents/Code/Litvak11/data/raw/subj1/off/R1.mat', 'data')

data_on = data_on.data
data_off = data_off.data

%% Original Sample Rate
for i = 1:9
    wave_off{i} = WaveletFilter(data_off.trial{1, 1}(i,:), 2400, 2400, freqs, [], [], []);
end


for i = 1:9
    wave_on{i} = WaveletFilter(data_on.trial{1, 1}(i,:), 2400, 2400, freqs, [], [], []);
end


for i = 1:9
        phase_off(i, :, :) = wave_off{i}.phase(:, :);
end

for i = 1:9
        amp_off(i, :, :) = wave_off{i}.wave(:, :);
end

for i = 1:9
        phase_on(i, :, :) = wave_on{i}.phase(:, :);
end

for i = 1:9
        amp_on(i, :, :) = wave_on{i}.wave(:, :);
end

time = wave_off{1,1}.time

% for i = 1:9
%         spectrum_on(i,:) = wave_on{1, i}.spectrum;
% end
% 
% for i = 1:9
%         spectrum_off(i,:) = wave_off{1, i}.spectrum;
% end

save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/freqs', 'freqs');
save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/time', 'time');

save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/wave_on', 'wave_on');
save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/phase_on', 'phase_on');
save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/amp_on', 'amp_on');

save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/wave_off', 'wave_off');
save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/phase_off', 'phase_off');
save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/amp_off', 'amp_off');

%save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/spectrum_off', 'spectrum_off');
%save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/spectrum_on', 'spectrum_on');





% 
% 
% %% Linear Freqs for fooof
% 
% freqs_lin = ([1:.5:256]);
% 
% 
% for i = 1:9
%     wave_off_lin{i} = WaveletFilter(data_off.trial{1, 1}(i,:), 2400, 2400, freqs_lin, [], [], []);
% end
% 
% 
% for i = 1:9
%     wave_on_lin{i} = WaveletFilter(data_on.trial{1, 1}(i,:), 2400, 2400, freqs_lin, [], [], []);
% end
% 
% 
% for i = 1:9
%         spectrum_on_lin(i,:) = wave_on_lin{1, i}.spectrum;
% end
% 
% for i = 1:9
%         spectrum_off_lin(i,:) = wave_off_lin{1, i}.spectrum;
% end
% 
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/freqs_lin', 'freqs_lin');
% 
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/spectrum_off_lin', 'spectrum_off_lin');
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/spectrum_on_lin', 'spectrum_on_lin');
% 
% 
% 
% 
% 
% %% Downsampling
% 
% for i = 1:9
%     wave_off_small{i} = WaveletFilter(data_off.trial{1, 1}(i,:), 2400, 500, freqs, [], [], []);
% end
% 
% 
% for i = 1:9
%     wave_on_small{i} = WaveletFilter(data_on.trial{1, 1}(i,:), 2400, 500, freqs, [], [], []);
% end
% 
% 
% for i = 1:9
%         phase_off_small(i, :, :) = wave_off_small{i}.phase(:, :);
% end
% 
% for i = 1:9
%         amp_off_small(i, :, :) = wave_off_small{i}.wave(:, :);
% end
% 
% for i = 1:9
%         phase_on_small(i, :, :) = wave_on_small{i}.phase(:, :);
% end
% 
% for i = 1:9
%         amp_on_small(i, :, :) = wave_on_small{i}.wave(:, :);
% end
% 
% time_small = wave_off_small{1,1}.time
% 
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/freqs', 'freqs');
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/time_small', 'time_small');
% 
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/wave_on_small', 'wave_on_small');
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/phase_on_small', 'phase_on_small');
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/on/amp_on_small', 'amp_on_small');
% 
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/wave_off_small', 'wave_off_small');
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/phase_off_small', 'phase_off_small');
% save('/Users/moritzgerster/Documents/Code/Litvak11/data/filtered/subj1/off/amp_off_small', 'amp_off_small');
% 


