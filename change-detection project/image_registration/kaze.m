function [align_gray,align_rgb,fixed,moving] = kaze(MOVING,FIXED,MOVING_RGB,threshold)
%registerImages  Register grayscale images using auto-generated code from Registration Estimator app.
%  [MOVINGREG] = registerImages(MOVING,FIXED) Register grayscale images
%  MOVING and FIXED using auto-generated code from the Registration
%  Estimator app. The values for all registration parameters were set
%  interactively in the app and result in the registered image stored in the
%  structure array MOVINGREG.

% Modified version based on Matlab Registration Estimator
% Code Source: Matlab Registration Estimator
% This program has two different threshold values. The program with 
% higher threshold value is used in precise image registration. The 
% program with lower threshold value is used in gif animination image
% registration.
%-----------------------------------------------------------


% Feature-based techniques require license to Computer Vision Toolbox
checkLicense()

% Default spatial referencing objects
fixedRefObj = imref2d(size(FIXED));
movingRefObj = imref2d(size(MOVING));

% Detect Kaze features
% Threshold value is tuned here to do the trade-off of running time and
% matched performance
fixedPoints = detectKAZEFeatures(FIXED,'Threshold',threshold,'NumOctaves',3,'NumScaleLevels',4);
movingPoints = detectKAZEFeatures(MOVING,'Threshold',threshold,'NumOctaves',3,'NumScaleLevels',4);

% Extract features
[fixedFeatures,fixedValidPoints] = extractFeatures(FIXED,fixedPoints,'Upright',false);
[movingFeatures,movingValidPoints] = extractFeatures(MOVING,movingPoints,'Upright',false);

% Match features
indexPairs = matchFeatures(fixedFeatures,movingFeatures,'MatchThreshold',50.000000,'MaxRatio',0.500000);
fixedMatchedPoints = fixedValidPoints(indexPairs(:,1));
movingMatchedPoints = movingValidPoints(indexPairs(:,2));
MOVINGREG.FixedMatchedFeatures = fixedMatchedPoints;
MOVINGREG.MovingMatchedFeatures = movingMatchedPoints;

% Apply transformation - Results may not be identical between runs because of the randomized nature of the algorithm
tform = estimateGeometricTransform(movingMatchedPoints,fixedMatchedPoints,'similarity');
MOVINGREG.Transformation = tform;
MOVINGREG.RegisteredImage = imwarp(MOVING_RGB, movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);
MOVINGREG.RegisteredImage_gray = imwarp(MOVING, movingRefObj, tform, 'OutputView', fixedRefObj, 'SmoothEdges', true);

% Store spatial referencing object
MOVINGREG.SpatialRefObj = fixedRefObj;

align_gray = MOVINGREG.RegisteredImage_gray;
align_rgb = MOVINGREG.RegisteredImage;
fixed = fixedMatchedPoints;
moving = movingMatchedPoints;

end

function checkLicense()

% Check for license to Computer Vision Toolbox
CVSTStatus = license('test','Video_and_Image_Blockset');
if ~CVSTStatus
    error(message('images:imageRegistration:CVSTRequired'));
end

end

