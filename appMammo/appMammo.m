%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                            = 1;
setting.db                              = path.db.ddsm;
setting.cnn                             = path.net.ddsm;
setting.neuralRegnDesc.layerId          = 19;
setting.neuralRegnDesc.maxSides         = [ 3000, 2500, 2000 ];
setting.neuralRegnDesc.regionFiltering  = ''; % 'imThr';
setting.neuralRegnDesc.pcaDim           = 128;
setting.neuralRegnDesc.kernelBeforePca  = 'NONE';
setting.neuralRegnDesc.normBeforePca    = 'L2';
setting.neuralRegnDesc.normAfterPca     = 'L2';
setting.neuralRegnDic.numTargetScale    = Inf;
setting.neuralRegnDic.numGaussian       = 256;
setting.fisher.normalizeByScale         = true;
setting.fisher.spatialPyramid           = '11';
setting.svm.kernel                      = 'NONE';
setting.svm.norm                        = 'L2';
setting.svm.c                           = 2; % 10;
setting.svm.epsilon                     = 1e-3;
setting.svm.biasMultiplier              = 1;
setting.svm.biasLearningRate            = 0.5;
setting.svm.loss                        = 'HINGE';
setting.svm.solver                      = 'SDCA';
setting.map.pre.scoreThrsh              = 0.0;
setting.map.pre.mapMaxSide              = 500;
setting.map.post.weightByIm             = true;
setting.map.post.smoothMap              = 5;
setting.map.post.smoothIm               = 10;
setting.map.post.cid2scaling            = [ 1.7; 1; 0 ];
setting.map.post.mapMaxVal              = 17;
setting.map.post.mapThrsh               = 5;
%% DO THE JOB.
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
db = db.mergeCls( { [ 1, 4 ]; [ 2, 5 ]; [ 3, 6 ]; }, 'DDSM_BCN' );
cnn = load( setting.cnn.path );
cnn.name = setting.cnn.name;
neuralRegnDscrber = ...
    NeuralRegnDscrber( db, cnn, ...
    setting.neuralRegnDesc, ...
    setting.neuralRegnDic );
neuralRegnDscrber.init( setting.gpus );
neuralRegnDscrber.trainDic;
% neuralRegnDscrber.descDb;
fisher = FisherMammo( neuralRegnDscrber, setting.fisher );
imDscrber = ImDscrber( db, { fisher }, [  ] );
imDscrber.descDb;
svm = Svm( db, imDscrber, setting.svm );
svm.trainSvm;
svm.evalSvm( 'visionresearchreport@gmail.com' );
map = Map( db, svm, setting.map.pre, setting.map.post );
map.genMapDb;





%% A TEST EXAMPLE AND VISUALIZATION.
close all; clearvars -except cnn db fisher imDscrber neuralRegnDscrber path setting svm map;
% Sample an image.
targetCid = 3;
iids = db.getTeiids;
iids = iids( cell2mat( db.iid2cids( iids ) ) == targetCid );
iid = randsample( iids, 1 ); % 4644; 4684; 2726;
oids = db.iid2oids{ iid };
im = imread( db.iid2impath{ iid } );
[ r, c, ~ ] = size( im );
% Set color and margin for display.
colorGt = [ 255, 255, 0 ];
colorBen = [ 0, 255, 0 ];
colorCan = [ 255, 0, 0 ];
margin = 30;
marginIm = zeros( size( im, 1 ), margin, 3, 'uint8' );
% Make a map.
mapdata = map.iid2map( iid );
% mapdata = map.im2map( im );
% Draw GT contour.
for n = 1 : numel( oids ),
    if ~isempty( db.oid2cont{ oids( n ) } )
        im = contourOnIm( im, db.oid2cont{ oids( n ) }, colorGt );
    end
end;
% Draw benign contour.
cid = 1;
maskBen = false( r, c );
for n = 1 : numel( mapdata.cid2cont{ cid } ),
    [ im, maskBen_ ] = contourOnIm( im, mapdata.cid2cont{ cid }{ n }, colorBen );
    maskBen = maskBen | maskBen_;
end;
% Draw cancer contour.
cid = 2;
maskCan = false( r, c );
for n = 1 : numel( mapdata.cid2cont{ cid } ),
    [ im, maskCan_ ] = contourOnIm( im, mapdata.cid2cont{ cid }{ n }, colorCan );
    maskCan = maskCan | maskCan_;
end;
% Make png.
mask = maskBen + maskCan;
imContOnly = cat( 3, ...
    maskBen * colorBen( 1 ) + maskCan * colorCan( 1 ), ...
    maskBen * colorBen( 2 ) + maskCan * colorCan( 2 ), ...
    maskBen * colorBen( 3 ) + maskCan * colorCan( 3 ) );
imContOnly = uint8( imContOnly );
% imwrite( imContOnly, 'test.png', 'png', 'Alpha', mask ); % <- This line for generating the png image with the alpha channel.
% Plot.
imshow( cat( 2, im, marginIm, mapdata.cid2map{ 1 }, marginIm, mapdata.cid2map{ 2 } ) );





