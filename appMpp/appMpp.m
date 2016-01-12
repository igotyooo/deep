%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.dstDir                          = '/nickel/data_mpp/';
setting.gpu                             = 1;
setting.db                              = path.db.voc2007;
setting.net                             = path.net.vgg_m;
setting.neuralRegnDesc.layerId          = 19;
setting.neuralRegnDesc.scalingCriteria  = 'MIN';
setting.neuralRegnDesc.maximumImageSize = 15e6; % To avoid memory overflow.
setting.neuralRegnDesc.pcaDim           = 128;
setting.neuralRegnDesc.kernelBeforePca  = 'NONE';
setting.neuralRegnDesc.normBeforePca    = 'L2';
setting.neuralRegnDesc.normAfterPca     = 'L2';
setting.neuralRegnDesc.regionFiltering  = '';
setting.neuralRegnDic.numTargetScale    = +Inf;
setting.neuralRegnDic.numGaussian       = 256;
setting.fisher.scaleWeightingMethod     = 'L2NORM'; % NONE, ENTROPY.
setting.fisher.spatialPyramid           = '11';
setting.svm.kernel                      = 'NONE';
setting.svm.norm                        = 'L2';
if strcmp( setting.db.name( 1 : 3 ), 'VOC' ),
setting.svm.c                           = 1;
else
setting.svm.c                           = 10;
end;
setting.svm.epsilon                     = 1e-3;
setting.svm.biasMultiplier              = 1;
setting.svm.biasLearningRate            = 0.5;
setting.svm.loss                        = 'HINGE';
setting.svm.solver                      = 'SDCA';

%% DO THE JOB.
reset( gpuDevice( setting.gpu ) );
db = Db( setting.db, setting.dstDir );
db.genDb;
net = load( setting.net.path );
net.name = setting.net.name;
patchSide = getNetProperties( net, setting.neuralRegnDesc.layerId );
setting.neuralRegnDesc.scaleId2numPixel = round( patchSide * 2 .^ ( 0 : 0.5 : 3 ) );
neuralRegnDscrber = NeuralRegnDscrber( db, net, setting.neuralRegnDesc, setting.neuralRegnDic );
neuralRegnDscrber.init( setting.gpu );
neuralRegnDscrber.trainDic;
neuralRegnDscrber.descDb;   % Possibly skipped.
setting.fisher.selectScales = 1 : numel( setting.neuralRegnDesc.scaleId2numPixel ) - 1;
fisher = Fisher( neuralRegnDscrber, setting.fisher );
imDscrber = ImDscrber( db, { fisher }, [  ] );
imDscrber.descDb;
svm = Svm( db, imDscrber, setting.svm );
svm.trainSvm;
svm.evalSvm( 'visionresearchreport@gmail.com' );