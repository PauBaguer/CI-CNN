clc
clear
close all

[inputs,targets,classnames] = load_input_dataset();
%Show first image
figure;
imshow(inputs{1})
[NImages,SizesImage,inputs_matrix,NChannels,Inputs_Net,Labels_Num,NLabels] = prepare_input(inputs, targets);

PCTtraining = 0.80;
PCTvalidation = 0.10;
PCTtest = 0.10;
[Inputs_Train,Labels_Train,Inputs_Valid,Labels_Valid,Inputs_Test,Labels_Test] = dataset_partition(Inputs_Net,Labels_Num,PCTtraining,PCTvalidation,PCTtest);