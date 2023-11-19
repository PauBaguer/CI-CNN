function [mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest,architecture,optimizationSolver)
%MEAN_ACCURACY Execute CNN K times to get a mean accuarcy measure
%   Detailed explanation goes here

Acc_Train_v = zeros(1,K);
Acc_Valid_v = zeros(1,K);
Acc_Test_v = zeros(1,K);
for i=1:K 
    [Acc_Train,Acc_Valid,Acc_Test] = exec_CNN(Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest,architecture,optimizationSolver);
    Acc_Train_v(i) = Acc_Train;
    Acc_Valid_v(i) = Acc_Valid;
    Acc_Test_v(i) = Acc_Test;
end

mean_Acc_Train = sum(Acc_Train_v) / K;
mean_Acc_Valid = sum(Acc_Valid_v) / K;
mean_Acc_Test = sum(Acc_Test_v) / K;
end

