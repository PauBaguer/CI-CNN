clc
clear
close all

K = 3;
PCTtraining = 0.80;
PCTvalidation = 0.10;
PCTtest = 0.10;


[inputs,targets,classnames] = load_input_dataset();
%Show first image
%figure;
%imshow(inputs{1})
[NImages,SizesImage,inputs_matrix,NChannels,Inputs_Net,Labels_Num,NLabels] = prepare_input(inputs, targets);
[Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest] = dataset_partition(Inputs_Net,Labels_Num,PCTtraining,PCTvalidation,PCTtest);




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
 'Plots', 'none'... %'training-progress' ... %Defalut value: 'none'
 );

[mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,PCTtraining,PCTvalidation,PCTtest,architecture,optimizationSolver);

fprintf('\n Final results \n');
fprintf('Training set Accuracy: %.2f%% \n',mean_Acc_Train*100);
fprintf('Validation set Accuracy: %.2f%% \n',mean_Acc_Valid*100);
fprintf('Test set Accuracy: %.2f%% \n',mean_Acc_Test*100);