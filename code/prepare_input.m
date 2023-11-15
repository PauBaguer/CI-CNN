function [NImages,SizesImage,inputs_matrix,NChannels,Inputs_Net,Labels_Num,NLabels] = prepare_input(inputs, targets)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
%% Inputs
% Store the total number of images and the sizes of the images (in this case all images have the same % size)
NImages = numel(inputs);
SizesImage = size(inputs{1});
% Converts the cell array of inputs to a 3D matrix
inputs_matrix = zeros(NImages,SizesImage(1),SizesImage(2));
for i=1:NImages
 inputs_matrix(i,:,:) = inputs{i};
end
NChannels = 1;

Inputs_Net = reshape(inputs_matrix,[SizesImage(1), SizesImage(2), NChannels, NImages]);
%% Targets
%Labels_1ofC = targets; % 1ofC coding scheme
Labels_Num = targets; % Numeric coding scheme
NLabels = numel(unique(Labels_Num));
%% Number of examples of every class
[Nc,~]=hist(Labels_Num,unique(Labels_Num));

figure;
histogram(Labels_Num,unique(Labels_Num)); %plot histogram
fprintf('Number of examples in every of the %d classes (%d):',numel(Nc),sum(Nc));
for n=Nc, fprintf(' %d',n); end; fprintf('\n');
end