clc;close;clear
%Load webcam addon
c=webcam;
%Load trained network
load myNet1;
%Set face detector with trained network to do facial recognition 
faceDetector=vision.CascadeObjectDetector;
while true
    e=c.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
     es=imcrop(e,bboxes(1,:));
    es=imresize(es,[227 227]);
    label=classify(myNet1,es);
    image(e);
    title(char(label));
    drawnow;
    %If no faces detected in previously captured image folders, show 'No Face Detected'
    else
        image(e);
        title('No Face Detected');
    end
end