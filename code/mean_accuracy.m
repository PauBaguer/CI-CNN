function [mean_Acc_Train,mean_Acc_Valid,mean_Acc_Test] = mean_accuracy(K,PCTtraining,PCTvalidation,PCTtest,architecture,optimizationSolver)
%MEAN_ACCURACY Execute CNN K times to get a mean accuarcy measure
%   Detailed explanation goes here

Acc_Train_v = zeros(k);
Acc_Valid_v = zeros(k);
Acc_Test_v = zeros(k);
for i=0:K 
    [Acc_Train,Acc_Valid,Acc_Test] = exec_CNN(PCTtraining,PCTvalidation,PCTtest,architecture,optimizationSolver);
    Acc_Train_v(i) = Acc_Train;
    Acc_Valid_v(i) = Acc_Valid;
    Acc_Test_v(i) = Acc_Test;
end

mean_Acc_Train = sum(Acc_Train_v) / K;
mean_Acc_Valid = sum(Acc_Valid_v) / K;
mean_Acc_Test = sum(Acc_Test_v) / K;
end

