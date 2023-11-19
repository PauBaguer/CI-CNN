function [Inputs_Train,Labels_Train,idxTrain,Inputs_Valid,Labels_Valid,idxValid,Inputs_Test,Labels_Test,idxTest] = dataset_partition(Inputs_Net,Labels_Num,PCTtraining,PCTvalidation,PCTtest)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

PCTnonTraining = PCTvalidation+PCTtest;
% For k-fold cross validation: cv = cvpartition(Labels_Num,'KFold',k,'Stratify',true);

cv = cvpartition(Labels_Num,'Holdout',PCTnonTraining,'Stratify',true); 
% For k-fold cross validation: idxTrain = training(cv,fold_number);
idxTrain = training(cv); 
% For k-fold cross validation: idxValid = test(cv,fold_number);
idxNonTrain = test(cv); 
% Obtain the inputs for training and validation from the indexes of cvpartition 
Inputs_Train = Inputs_Net(:,:,:,idxTrain);
Labels_Train = categorical(Labels_Num(idxTrain)); % Labels for the network must be categorical

Inputs_NonTrain = Inputs_Net(:,:,:,idxNonTrain);
Labels_Num_NonTrain = Labels_Num(idxNonTrain);


relative_PCTtest = PCTtest/(PCTvalidation+PCTtest);
cv = cvpartition(Labels_Num_NonTrain,'Holdout',relative_PCTtest,'Stratify',true);
idxValid = training(cv); 
idxTest = test(cv);

Inputs_Valid = Inputs_NonTrain(:,:,:,idxValid);
Labels_Valid = categorical(Labels_Num_NonTrain(idxValid)); % Labels for the network must be categorical

Inputs_Test = Inputs_NonTrain(:,:,:,idxTest);
Labels_Test = categorical(Labels_Num_NonTrain(idxTest));


% Statistics of the partition
fprintf('Number of examples in the training set: %d\n',size(Inputs_Train,4));
fprintf('Number of examples in the validation set: %d\n',size(Inputs_Valid,4));
fprintf('Number of examples in the test set: %d\n',size(Inputs_Test,4));


[Nc,~]=hist(Labels_Num,unique(Labels_Num));
figure;
subplot(2,2,1);
histogram(Labels_Num,unique(Labels_Num)); %plot histogram
title('Complete dataset histogram');
fprintf('Number of examples in every of the %d classes (%d):',numel(Nc),sum(Nc));
for n=Nc, fprintf(' %d',n); end; fprintf('\n');

[Nc,~]=hist(Labels_Train,unique(Labels_Train));
subplot(2,2,2);
histogram(double(Labels_Train),unique(double(Labels_Train))); %plot histogram
title('Training set histogram')
fprintf('Number of examples in validation set in every of the %d classes (%d):',numel(Nc),sum(Nc));
for n=Nc, fprintf(' %d',n); end; fprintf('\n');

[Nc,~]=hist(Labels_Valid,unique(Labels_Valid));
subplot(2,2,3);
histogram(double(Labels_Valid),unique(double(Labels_Valid))); %plot shistogram
title('Validation set histogram')
fprintf('Number of examples in test set in every of the %d classes (%d):',numel(Nc),sum(Nc));
for n=Nc, fprintf(' %d',n); end; fprintf('\n');

[Nc,~]=hist(Labels_Test,unique(Labels_Test));
subplot(2,2,4);
histogram(double(Labels_Test),unique(double(Labels_Test))); %plot histogram
title('Test set histogram')
fprintf('Number of examples in training set in every of the %d classes (%d):',numel(Nc),sum(Nc));
for n=Nc, fprintf(' %d',n); end; fprintf('\n');

end