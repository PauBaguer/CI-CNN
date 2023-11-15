function [inputs,targets,classnames] = load_input_dataset()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
mat = load('..\caltech101_silhouettes_28.mat');
s = size(mat.X);
inputs = cell(s(1),1);
for i=1:s(1)
    inputs{i} = reshape(mat.X(i,:),[28,28]);
targets = mat.Y;
classnames = mat.classnames;
end