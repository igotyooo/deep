%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
gpuDevice( 1 ); reset( gpuDevice );
addpath( genpath( '..' ) ); init;
setting.useGpu                          = true;
setting.db                              = path.db.ddsm;
setting.cnn                             = path.net.ddsm;
setting.neuralRegnDesc.layerId          = 19;
setting.neuralRegnDesc.maxSides         = [ 3000, 2500, 2000 ];
setting.neuralRegnDesc.pcaDim           = 128;
setting.neuralRegnDesc.kernelBeforePca  = 'NONE';
setting.neuralRegnDesc.normBeforePca    = 'L2';
setting.neuralRegnDesc.normAfterPca     = 'L2';
setting.neuralRegnDic.numTargetScale    = Inf;
setting.neuralRegnDic.numGaussian       = 256;
setting.fisher.normalizeByScale         = true;
setting.fisher.spatialPyramid           = '11';
setting.fisher.regionFiltering          = '';
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
setting.map.post.mapMaxVal              = 50;
%% DO THE JOB.
db = Db( setting.db, path.dstDir );
db.genDb;
db = db.mergeCls( { [ 1, 4 ]; [ 2, 5 ]; [ 3, 6 ]; }, 'DDSM_BCN' );
% db = db.mergeCls( { [ 1, 2, 4, 5 ]; [ 3, 6 ]; }, 'DDSM_MN' );
cnn = load( setting.cnn.path );
cnn.name = setting.cnn.name;
neuralRegnDscrber = ...
    NeuralRegnDscrberMammo( db, cnn, ...
    setting.neuralRegnDesc, ...
    setting.neuralRegnDic, setting.useGpu );
neuralRegnDscrber.init;
neuralRegnDscrber.trainDic;
neuralRegnDscrber.descDb;
fisher = FisherMammo( neuralRegnDscrber, setting.fisher );
imDscrber = ImDscrber( db, { fisher }, [  ] );
imDscrber.descDb;
svm = SvmBch( db, imDscrber, setting.svm );
svm.trainSvm;
svm.evalSvm( 'visionresearchreport@gmail.com' );






%% DEV MAP.
close all; clearvars -except cnn db fisher imDscrber neuralRegnDscrber path setting svm;
iids = db.getTeiids;
iids = iids( cell2mat( db.iid2cids( iids ) ) == 2 );
iid = randsample( iids, 1 ); % 4644; 4684; 2726;
cid = db.iid2cids{ iid };
cname = db.cid2name{ cid };
margin = 10;
im = imread( db.iid2impath{ iid } );
im = contourOnIm( im, db.oid2cont{ iid }, [ 255, 255, 0 ] );
marginIm = zeros( size( im, 1 ), margin, 3, 'uint8' );

map = Map( db, svm, setting.map.pre, setting.map.post );
cid2map = map.iid2map( iid );

imshow( cat( 2, im, marginIm, cid2map{ 1 }, marginIm, cid2map{ 2 } ) );
title( sprintf( 'GT: %s', cname ) );














