clc
clear all
close all
warning off
%AlexNet is a pretrained Convolutional Neural Network (CNN) that has been trained on approximately 1.2 million images from the ImageNet Dataset
g=alexnet;
%Set layers
layers=g.Layers;
layers(23)=fullyConnectedLayer(2);
layers(25)=classificationLayer;
%Set collected images folder
allImages=imageDatastore('Faces','IncludeSubfolders',true, 'LabelSource','foldernames');
%Implementing stochastic gradient descent with momentum (SGDM) algorithm to applies the SGDM optimization algorithm to update network parameters in custom training loops
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
%Training network
myNet1=trainNetwork(allImages,layers,opts);
save myNet1;