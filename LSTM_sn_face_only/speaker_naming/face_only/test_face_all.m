addpath utils/

clearvars -global config;
global config mem;
gpuDevice(1);
sn_face_init();

load('speaker_naming/results/face_2/61.mat');
config.weights = model.weights;
config.data_mean = model.data_mean;
config.one_over_data_std = model.one_over_data_std;

load('data/speaker_naming/val_face/1');
% test_samples = test_samples(:,:,1:2000);
% test_labels = test_labels(:,1:2000);
test_labels = reshape(test_labels, size(test_labels,1), 1, size(test_labels,2));
test_labels = repmat(test_labels, [1 config.max_time_steps 1]);
test_samples = config.NEW_MEM(test_samples);
test_labels = config.NEW_MEM(test_labels);
test_samples = bsxfun(@times, bsxfun(@minus, test_samples, config.data_mean), config.one_over_data_std);

val_cost = 0;
correct_num = 0;
train_cost = 0;
train_correct_num = 0;
for ii = 1:size(test_samples, 3)/config.batch_size
    fprintf('%d/%d\n', ii * config.batch_size, floor(size(test_samples, 3)/config.batch_size)*config.batch_size);
    start_idx = config.batch_size * (ii-1) + 1;
    end_idx = start_idx + config.batch_size - 1;

    val_sample = test_samples(:,:,start_idx:end_idx);
    val_label = test_labels(:,:,start_idx:end_idx);                    

    lstm_forward_v4(val_sample, val_label);                    
    val_cost = val_cost + config.cost;

    [value, estimated_labels] = max(mem.net_out(:,end,:));
    [value, true_labels] = max(val_label(:,end,:));
    correct_num = correct_num + length(find(estimated_labels == true_labels));
end

val_cost = val_cost / (size(test_samples, 3)/config.batch_size);
acc = correct_num / double(floor(size(test_samples, 3)/config.batch_size)*config.batch_size);
fprintf('\nval cost: %f, val_acc: %.2f%%\n', val_cost, acc*100);
