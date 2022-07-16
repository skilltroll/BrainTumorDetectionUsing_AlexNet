clear; close all;clc
%DATASET LOCATION AND CONFIGURATION 
imds = imageDatastore("C:\Users\itskilltroll\Desktop\gradproject","IncludeSubfolders",true,"LabelSource","foldernames");

[imdsTrain, imdsTest] = splitEachLabel(imds,0.7);

% Resize the images to match the network input layer.
augimdsTrain = augmentedImageDatastore([227 227 3],imdsTrain);
augimdsTest = augmentedImageDatastore([227 227 3],imdsTest);

opts = trainingOptions("adam",...
    'MiniBatchSize',50,...
    'MaxEpochs',160, ...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.0004,...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "ValidationData",augimdsTest,...
    'ValidationFrequency',50);

% ALEXNET LAYERS 
layers = [
    imageInputLayer([227 227 3])
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu3")
    groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu4")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu5")
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(2,"Name","fc8_New")
    softmaxLayer("Name","prob")
    classificationLayer("Name","output_new")];
% END OF ALEXNET LAYERS SECTION 
[net, traininfo] = trainNetwork(augimdsTrain,layers,opts);
%training progress fig -menubar / fig will be saved 
fig = findall(groot,'Type','figure');
fig.MenuBar = 'figure';


[YPred, scores] = classify(net,augimdsTest);
YValidation = imdsTest.Labels;

accuracy = mean(YPred == YValidation);

% construct the confusion matrix
figure 
cm = confusionchart(YValidation,YPred);
cm.Normalization = 'total-normalized';
cm.Title = 'Confusion Matrix';
aa=confusionmat(YValidation,YPred);
savefig('fig-N-ABN.fig')

% construct the AUC for the obtained result
YValidation2=double(nominal(imdsTest.Labels));
[X,Y,T,AUC] = perfcurve(YValidation2,scores(:,1),1);

% fig 1 AUC
figure
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('AUC for Classification by Modified AlexNet')
cd 'C:\Users\itskilltroll\Desktop\gradproject'
save('reuseme.mat')
savefig('fig-AUC.fig')

