%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
gpuDevice( 1 ); reset( gpuDevice );
addpath( genpath( '..' ) ); init;
setting.useGpu                          = true;
setting.db                              = path.db.ddsm;
setting.cnn                             = path.net.ddsm;
setting.neuralRegnDesc.layerId          = 19;
setting.neuralRegnDesc.maxSides         = [ 2400, 2100, 1800 ];
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

%% DO THE JOB.
db = Db( setting.db, path.dstDir );
db.genDb;
cnn = load( setting.cnn.path );
cnn.name = setting.cnn.name;
neuralRegnDscrber = ...
    NeuralRegnDscrberMammo( db, cnn, ...
    setting.neuralRegnDesc, ...
    setting.neuralRegnDic, setting.useGpu );
neuralRegnDscrber.init;
neuralRegnDscrber.trainDic;



%% 



















fisher = Fisher( neuralRegnDscrber, setting.fisher );
imDscrber = ImDscrber( db, { fisher }, [  ] );
imDscrber.descDb;
svm = SvmBch( db, imDscrber, setting.svm );
svm.setPrll( false );
svm.trainSvm;
svm.evalSvm( 'visionresearchreport@gmail.com' );














