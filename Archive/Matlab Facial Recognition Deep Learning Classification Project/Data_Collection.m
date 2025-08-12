clc
clear all
close all
warning off;
%Use webcam addon and choose face detector
cao=webcam;
faceDetector=vision.CascadeObjectDetector;
%Capture 150 images with while loop
c=150;
temp=0;
while true
    e=cao.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
    if(temp>=c)
        break;
    %Crop images and resize them to 227, 227 for alexnet to train model
    else
    es=imcrop(e,bboxes(1,:));
    es=imresize(es,[227 227]);
    %Set image filetype and convert number to string to save as filename
    filename=strcat(num2str(temp),'.bmp');
    imwrite(es,filename);
    temp=temp+1;
    imshow(es);
    drawnow;
    end
    else
        imshow(e);
        drawnow;
    end
end