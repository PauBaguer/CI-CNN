clc
clear
close all

K = 3;
PCTtraining = 0.80;
PCTvalidation = 0.10;
PCTtest = 0.10;
NB = 1; % Number of convolutional blocks

[inputs,targets,classnames] = load_input_dataset();
%Show first image
figure;
imshow(inputs{1})
[NImages,SizesImage,inputs_matrix,NChannels,Inputs_Net,Labels_Num,NLabels] = prepare_input(inputs, targets);
[Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest] = dataset_partition(Inputs_Net,Labels_Num,PCTtraining,PCTvalidation,PCTtest);


% Define architecture for CNN with one convolutional block (FS = 128)
net_one_block = [
    imageInputLayer([SizesImage(1), SizesImage(2), NChannels])
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    flattenLayer
    fullyConnectedLayer(NLabels)
    softmaxLayer
    classificationLayer
];

% Optimize learningRateRange
%learningRateRange = logspace(-6,0,14);
% learningRateRange = logspace(-3,-1,15);
%learningRateRange = linspace(0.003,0.007,15);

learningRateRange = logspace(-5,-3,14);
results = zeros(length(learningRateRange),3);
parfor i = 1:length(learningRateRange)

    optimizationSolver = ...
     trainingOptions('adam', ... %'sgdm' | 'rmsprop' | 'adam' | 'lbfgs'
     'InitialLearnRate', learningRateRange(i), ... %Defalut value: 0.01
     'GradientDecayFactor', 0.9, ... %Default value: 0.9
     'SquaredGradientDecayFactor', 0.999, ... % Default value: 0.999
     'Epsilon', 1e-8, ... % Default value: 1e-8
     ... %'L2Regularization', 1, ... %Defalut value: 1.0000e-0
     'MaxEpochs', 50, ... %Defalut value: 30
     'MiniBatchSize', 32, ... %Defalut value: 128
     'Verbose', 0, ... %Defalut value: 1
     'ValidationData', {Inputs_Valid,Labels_Valid}, ... %Defalut value: []
     'ValidationFrequency', 50, ... %Defalut value: 50
     'ValidationPatience', 5, ... %Defalut value: 5
     'Shuffle', 'once', ... %Defalut value: 'once'
     'Plots', 'none' ... %Defalut value: 'none'
     );



[mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest,net_one_block,optimizationSolver);

fprintf('\n Final results for learning rate: %d\n',learningRateRange(i));
fprintf('Training set Accuracy: %.2f%% \n',mean_Acc_Train*100);
fprintf('Validation set Accuracy: %.2f%% \n',mean_Acc_Valid*100);
fprintf('Test set Accuracy: %.2f%% \n',mean_Acc_Test*100);
results(i,:) = [mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test];
end

f = figure;
semilogx(learningRateRange,results);
%plot(learningRateRange,results);
legend('Training','Validation','Test');
title('Learning rate optimization');
xlabel('Learning rate');
ylabel('Accuracy');

name=sprintf('figs/learningrate-%s.png',datestr(now,'HH-MM-SS'));
saveas(f,name,'png');


% Learning rate = 0.005




% L2RegularizationRange = logspace(-6,0,14);
% L2RegularizationRange = logspace(-3,-1,15);
% 
% results = zeros(length(L2RegularizationRange),3);
% parfor i = 1:length(L2RegularizationRange)
% 
%     optimizationSolver = ...
%      trainingOptions('adam', ... %'sgdm' | 'rmsprop' | 'adam' | 'lbfgs'
%      'InitialLearnRate', 0.005, ... %Defalut value: 0.01
%      'GradientDecayFactor', 0.9, ... %Default value: 0.9
%      'SquaredGradientDecayFactor', 0.999, ... % Default value: 0.999
%      'Epsilon', 1e-8, ... % Default value: 1e-8
%      'L2Regularization', L2RegularizationRange(i), ... %Defalut value: 1.0000e-0
%      'MaxEpochs', 20, ... %Defalut value: 30
%      'MiniBatchSize', 128, ... %Defalut value: 128
%      'Verbose', 0, ... %Defalut value: 1
%      'ValidationData', {Inputs_Valid,Labels_Valid}, ... %Defalut value: []
%      'ValidationFrequency', 10, ... %Defalut value: 50
%      'ValidationPatience', inf, ... %Defalut value: 5
%      'Shuffle', 'every-epoch', ... %Defalut value: 'once'
%      'Plots', 'none', ... %Defalut value: 'none'
%      'OutputNetwork', 'best-validation-loss' ...
%      );
% 
%  %'Momentum', 0.9, ... %Defalut value: 0.9000
% 
% 
% 
% [mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest,net_one_block,optimizationSolver);
% 
% fprintf('\n Final results L2 Regularization index: %d\n',L2RegularizationRange(i));
% fprintf('Training set Accuracy: %.2f%% \n',mean_Acc_Train*100);
% fprintf('Validation set Accuracy: %.2f%% \n',mean_Acc_Valid*100);
% fprintf('Test set Accuracy: %.2f%% \n',mean_Acc_Test*100);
% 
% results(i,:) = [mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test];
% end
% 
% f = figure;
% semilogx(L2RegularizationRange,results);
% %plot(L2RegularizationRange,results);
% legend('Training','Validation','Test');
% title('L2 regularization optimization');
% xlabel('L2 regularization index');
% ylabel('Accuracy');
% 
% name=sprintf('figs/l2regularization-%s.png',datestr(now,'HH-MM-SS'));
% saveas(f,name,'png');

% Best L2 regularization: 0.0072




%GradientDecayFactor = logspace(-3,0,10) * 0.9;
% GradientDecayFactor = linspace(0.002,0.005,15);
% results = zeros(length(GradientDecayFactor),3);
% parfor i = 1:length(GradientDecayFactor)
% 
%     optimizationSolver = ...
%      trainingOptions('adam', ... %'sgdm' | 'rmsprop' | 'adam' | 'lbfgs'
%      'InitialLearnRate', 0.005, ... %Defalut value: 0.01
%      'GradientDecayFactor', GradientDecayFactor(i), ... %Default value: 0.9
%      'SquaredGradientDecayFactor', 0.999, ... % Default value: 0.999
%      'Epsilon', 1e-8, ... % Default value: 1e-8
%      'L2Regularization', 0.0072, ... %Defalut value: 1.0000e-0
%      'MaxEpochs', 20, ... %Defalut value: 30
%      'MiniBatchSize', 128, ... %Defalut value: 128
%      'Verbose', 0, ... %Defalut value: 1
%      'ValidationData', {Inputs_Valid,Labels_Valid}, ... %Defalut value: []
%      'ValidationFrequency', 10, ... %Defalut value: 50
%      'ValidationPatience', inf, ... %Defalut value: 5
%      'Shuffle', 'every-epoch', ... %Defalut value: 'once'
%      'Plots', 'none', ... %Defalut value: 'none'
%      'OutputNetwork', 'best-validation-loss' ...
%      );
% 
%  %'Momentum', 0.9, ... %Defalut value: 0.9000
% 
% 
% 
% [mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest,net_one_block,optimizationSolver);
% 
% fprintf('\n Final results Gradient Decay Factor index: %d\n',GradientDecayFactor(i));
% fprintf('Training set Accuracy: %.2f%% \n',mean_Acc_Train*100);
% fprintf('Validation set Accuracy: %.2f%% \n',mean_Acc_Valid*100);
% fprintf('Test set Accuracy: %.2f%% \n',mean_Acc_Test*100);
% 
% results(i,:) = [mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test];
% end
% 
% f = figure;
% plot(GradientDecayFactor,results);
% %plot(L2RegularizationRange,results);
% legend('Training','Validation','Test');
% title('Gradient Decay Factor optimization');
% xlabel('Gradient Decay Factor');
% ylabel('Accuracy');
% 
% name=sprintf('figs/GradientDecayFactor-%s.png',datestr(now,'HH-MM-SS'));
% saveas(f,name,'png');

% Best GradientDecayFactor: 0.004


% Define architecture for CNN with three convolutional blocks (FS = 32, 64, 128)
% net = [
%     imageInputLayer([SizesImage(1), SizesImage(2), NChannels])
% ];

% filter_sizes = [32, 64, 128];
% for i = 1:3
%     net = [
%         net
%         convolution2dLayer(3, filter_sizes(i), 'Padding', 'same')
%         sigmoidLayer
%         maxPooling2dLayer(2, 'Stride', 2)
%     ];
% end
% 
% net = [
%     net
%     fullyConnectedLayer(NLabels)
%     softmaxLayer
%     classificationLayer
% ];
% 


