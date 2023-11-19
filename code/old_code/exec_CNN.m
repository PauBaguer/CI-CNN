function [Acc_Train,Acc_Valid,Acc_Test] = exec_CNN(Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest,architecture,optimizationSolver)
%EXEC_CNN Execute one iteration of the CNN
%   Detailed explanation goes here


net = trainNetwork(Inputs_Train,Labels_Train,architecture,optimizationSolver);
[Acc_Train,Acc_Valid,Acc_Test] = validation(net,Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest);
end

