function [Acc_Train,Acc_Valid,Acc_Test] = validation(net,Inputs_Train,Labels_Train,Inputs_Valid,Labels_Valid,Inputs_Test,Labels_Test)
%VALIDATION Summary of this function goes here
%   Detailed explanation goes here


Predictions_Train = predict(net,Inputs_Train);
Predictions_Valid = predict(net,Inputs_Valid);
Predictions_Test = predict(net,Inputs_Test);

Labels_Num_Train = double(Labels_Train);
Labels_Num_Valid = double(Labels_Valid);
Labels_Num_Test = double(Labels_Test);
% Accuracies
Acc_Train = sum(vec2ind(Predictions_Train') == Labels_Num_Train) / sum(idxTrain);
Acc_Valid = sum(vec2ind(Predictions_Valid') == Labels_Num_Valid) / sum(idxValid);
Acc_Test = sum(vec2ind(Predictions_Test') == Labels_Num_Test) / sum(idxTest);

fprintf('Training set Accuracy: %d\n',Acc_Train);
fprintf('Validation set Accuracy: %d\n',Acc_Valid);
fprintf('Test set Accuracy: %d\n',Acc_Test);

end

