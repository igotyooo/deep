%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpu                             = 2;
setting.db                              = path.db.indoor_devices;
setting.net                             = path.net.vgg_m;
setting.neuralRegnDesc.layerId          = 19;
setting.neuralRegnDesc.scalingCriteria  = 'MIN';
setting.neuralRegnDesc.scaleId2numPixel = round( 227 * 2 .^ ( 0 : 0.5 : 3 ) );
setting.neuralRegnDesc.pcaDim           = 128;
setting.neuralRegnDesc.kernelBeforePca  = 'NONE';
setting.neuralRegnDesc.normBeforePca    = 'L2';
setting.neuralRegnDesc.normAfterPca     = 'L2';
setting.neuralRegnDesc.regionFiltering  = '';
setting.neuralRegnDic.numTargetScale    = +Inf;
setting.neuralRegnDic.numGaussian       = 256;
setting.fisher.selectScales             = 1 : numel( setting.neuralRegnDesc.scaleId2numPixel ) - 4;
setting.fisher.scaleWeightingMethod     = 'NONE'; % NONE, ENTROPY.
setting.fisher.spatialPyramid           = '11';
setting.neuralDesc.layerId              = 19;
setting.neuralDesc.augmentationType     = 'NONE';
setting.imDscrber.weights               = [ 1; 0.6; ];
setting.svm.kernel                      = 'NONE';
setting.svm.norm                        = 'L2';
setting.svm.c                           = 10;
setting.svm.epsilon                     = 1e-3;
setting.svm.biasMultiplier              = 1;
setting.svm.biasLearningRate            = 0.5;
setting.svm.loss                        = 'HINGE';
setting.svm.solver                      = 'SDCA';


%% DO THE JOB.
reset( gpuDevice( setting.gpu ) );
db = Db( setting.db, path.dstDir );
db.genDb;
net = load( setting.net.path );
net.name = setting.net.name;
neuralRegnDscrber = ...
    NeuralRegnDscrber( db, net, ...
    setting.neuralRegnDesc, ...
    setting.neuralRegnDic );
neuralRegnDscrber.init( setting.gpu );
neuralRegnDscrber.trainDic;
neuralRegnDscrber.descDb;   % Possibly skipped.
fisher = Fisher( neuralRegnDscrber, setting.fisher );
neuralDesc = NeuralDscrber( db, net, setting.neuralDesc );
neuralDesc.init( setting.gpu );
imDscrber = ImDscrber( db, { fisher; neuralDesc; }, setting.imDscrber );
imDscrber.descDb;
svm = Svm( db, imDscrber, setting.svm );
svm.trainSvm;
svm.evalSvm( 'visionresearchreport@gmail.com' );

%% RESULT: CONFUSION METRIX.
clc; close all;
clearvars -except db fisher imDscrber net neuralDesc neuralRegnDscrber path setting svm;
confMat = svm.result.confMat;
imagesc( confMat );  colorbar;
set( gcf, 'color', 'w' );

%% RESULT: EXAMPLES.
clc; clearvars -except db fisher imDscrber net neuralDesc neuralRegnDscrber path setting svm;
cid = 7;
topn = 3;
select = false;

[ ~, rank2idx2cid ] = sort( svm.cid2teidx2score, 'descend' );
idx2target = db.oid2cid( svm.teidx2iid )' == cid;
idx2true = rank2idx2cid( 1, : ) == cid;
trueidxs = find( idx2target & idx2true );
falseidxs = find( idx2target & ~idx2true );
if select, 
    idx = trueidxs( ceil( numel( trueidxs ) * rand ) );
else
    idx = falseidxs( ceil( numel( falseidxs ) * rand ) );
end;
iid = svm.teidx2iid( idx );
[ rank2score, rank2cid ] = sort( svm.cid2teidx2score( :, idx ), 'descend' );
im = imread( db.iid2impath{ iid } );
subplot( 1, 2, 1 ); imshow( im );
subplot( 1, 2, 2 ); bar( rank2score( 1 : topn ) );
cid2name = { 'AIR'; 'LI(OFF)'; 'LI(ON)'; 'REF'; 'TV(OFF)'; 'TV(ON)'; 'WASH' };
set( gca, 'XTick', 1 : topn, 'XTickLabel', cid2name( rank2cid( 1 : topn ) ) );
set( gcf, 'Color', 'w' );

%% RESULT: SPEED.
numIm = 10;
t = 0;
for iid = 1 : numIm;
    im = imread( db.iid2impath{ iid } );
    it = tic;
    svm.predictIm( im );
    t = t + toc( it );
end;
t / numIm;


