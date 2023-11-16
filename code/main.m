clc
clear
close all



PCTtraining = 0.80;
PCTvalidation = 0.10;
PCTtest = 0.10;

architecture = [
 imageInputLayer([SizesImage(1), SizesImage(2), NChannels])
 convolution2dLayer(3,128,'Padding','same') %KernelSize, NumberOfKernels, rest of parameters
 reluLayer
 maxPooling2dLayer(2,'Stride',2) %PoolingSize, rest of parameters
 fullyConnectedLayer(NLabels)
 softmaxLayer
 classificationLayer
];

optimizationSolver = ...
 trainingOptions('adam', ... %'sgdm' | 'rmsprop' | 'adam' | 'lbfgs'
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

[mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,PCTtraining,PCTvalidation,PCTtest,architecture,optimizationSolver);

