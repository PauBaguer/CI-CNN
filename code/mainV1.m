clc
clear
close all

K = 3;
PCTtraining = 0.80;
PCTvalidation = 0.10;
PCTtest = 0.10;
NB = 1; % Number of convolutional blocks

[inputs,targets,classnames] = load_input_dataset();
% Show first image
% figure;
% imshow(inputs{1})
[NImages,SizesImage,inputs_matrix,NChannels,Inputs_Net,Labels_Num,NLabels] = prepare_input(inputs, targets);
[Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest] = dataset_partition(Inputs_Net,Labels_Num,PCTtraining,PCTvalidation,PCTtest);

% Define architecture for CNN with one convolutional block (FS = 128)
net_one_block = [
    imageInputLayer([SizesImage(1), SizesImage(2), NChannels])
    convolution2dLayer(3, 128, 'Padding', 'same')
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(NLabels)
    softmaxLayer
    classificationLayer
];

% Define architecture for CNN with three convolutional blocks (FS = 32, 64, 128)
net = [
    imageInputLayer([SizesImage(1), SizesImage(2), NChannels])
];

filter_sizes = [32, 64, 128];
for i = 1:3
    net = [
        net
        convolution2dLayer(3, filter_sizes(i), 'Padding', 'same')
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
    ];
end

net = [
    net
    fullyConnectedLayer(NLabels)
    softmaxLayer
    classificationLayer
];


optimizationSolver = ...
 trainingOptions('sgdm', ... %'sgdm' | 'rmsprop' | 'adam' | 'lbfgs'
 'InitialLearnRate', 0.01, ... %Defalut value: 0.01
 'Momentum', 0.9, ... %Defalut value: 0.9000
 'L2Regularization', 1.0000e-04, ... %Defalut value: 1.0000e-0
 'MaxEpochs', 30, ... %Defalut value: 30
 'MiniBatchSize', 128, ... %Defalut value: 128
 'Verbose', 1, ... %Defalut value: 1
 'ValidationData', {Inputs_Valid,Labels_Valid}, ... %Defalut value: []
 'ValidationFrequency', 50, ... %Defalut value: 50
 'ValidationPatience', 5, ... %Defalut value: 5
 'Shuffle', 'once', ... %Defalut value: 'once'
 'Plots', 'training-progress' ... %Defalut value: 'none'
 );

[mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,PCTtraining,PCTvalidation,PCTtest,net,optimizationSolver);
