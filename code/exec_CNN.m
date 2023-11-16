function [Acc_Train,Acc_Valid,Acc_Test] = exec_CNN(PCTtraining,PCTvalidation,PCTtest,architecture,optimizationSolver)
%EXEC_CNN Execute one iteration of the CNN
%   Detailed explanation goes here

[inputs,targets,classnames] = load_input_dataset();
[NImages,SizesImage,inputs_matrix,NChannels,Inputs_Net,Labels_Num,NLabels] = prepare_input(inputs, targets);
[Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest] = dataset_partition(Inputs_Net,Labels_Num,PCTtraining,PCTvalidation,PCTtest);
net = trainNetwork(Inputs_Train,Labels_Train,architecture,optimizationSolver);
[Acc_Train,Acc_Valid,Acc_Test] = validation(net,Inputs_Train,Labels_Train,Inputs_Valid,Labels_Valid,Inputs_Test,Labels_Test);
end

