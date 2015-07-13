clc; clear all; close all;
gpu = 1;
reset( gpuDevice( gpu ) );
addpath( genpath( '..' ) ); init;
% Load net.
netPathBeta7 = '/nickel/data_newdb/VOC2007/CNN_LR0000067122_STFRM_PTVGGM_OF_IODETSINGLECLS_BS128_OF_DBTS_FOM1P5_OF_DB/E0010.mat';
net_ = load( netPathBeta7 );
net_ = net_.cnn;
% Do the job.
net_ = vl_simplenn_move( net_, 'cpu' );
net_.avgIm = gather( net_.avgIm );
numLyr = numel( net_.layers );
for lid = 1 : numLyr,
    layer = net_.layers{ lid };
    if ~isfield( layer, 'filters' ), continue; end;
    layer.weights = cell( 1, 2 );
    layer.weights{ 1 } = layer.filters;
    layer.weights{ 2 } = layer.biases;
    layer = rmfield( layer, 'filters' );
    layer = rmfield( layer, 'biases' );
    if isfield( layer, 'filtersMomentum' ), 
        layer.momentum = cell( 1, 2 );
        layer.momentum{ 1 } = layer.filtersMomentum;
        layer.momentum{ 2 } = layer.biasesMomentum;
    end; 
    layer = rmfield( layer, 'filtersMomentum' );
    layer = rmfield( layer, 'biasesMomentum' );
    if isfield( layer, 'filtersLearningRate' ), 
        layer.learningRate = [ layer.filtersLearningRate, layer.biasesLearningRate ];
    end;
    layer = rmfield( layer, 'filtersLearningRate' );
    layer = rmfield( layer, 'biasesLearningRate' );
    if isfield( layer, 'filtersWeightDecay' ), 
        layer.weightDecay = [ layer.filtersWeightDecay, layer.biasesWeightDecay ];
    end;
    layer = rmfield( layer, 'filtersWeightDecay' );
    layer = rmfield( layer, 'biasesWeightDecay' );
    net_.layers{ lid } = layer;
end;

%% SET PARAMETERS ONLY.
setting.db                                          = path.db.voc2007;
setting.gpus                                        = gpu; % [ 1, 2 ];
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.dstSide                             = 227;
setting.io.tsDb.numScale                            = 8;
setting.io.tsDb.scaleStep                           = 2;
setting.io.tsDb.startHorzScale                      = 1.5;
setting.io.tsDb.horzScaleStep                       = 0.5;
setting.io.tsDb.endHorzScale                        = 4;
setting.io.tsDb.startVertScale                      = 1.5;
setting.io.tsDb.vertScaleStep                       = 0.5;
setting.io.tsDb.endVertScale                        = 2;
setting.io.tsDb.insectOverFgdObj                    = 0.5;
setting.io.tsDb.insectOverFgdObjForMajority         = 0.1;
setting.io.tsDb.fgdObjMajority                      = 1.5;
setting.io.tsDb.insectOverBgdObj                    = 0.2;
setting.io.tsDb.insectOverBgdRegn                   = 0.5;
setting.io.tsDb.insectOverBgdRegnForReject          = 0.9;
setting.io.tsDb.numDirection                        = 3;
setting.io.tsDb.numMaxBgdRegnPerScale               = 100;
setting.io.tsDb.stopSignError                       = 5;
setting.io.tsDb.minObjScale                         = 1 / sqrt( 2 );
setting.io.tsDb.numErode                            = 5;
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 10;
setting.io.tsNet.outFilterDepth                     = 2048;
setting.io.general.dstSide                          = 227;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128 * numel( setting.gpus );
setting.net.normalizeImage                          = 'AVGIM'; % 'RGBMEAN';
setting.net.weightDecay                             = 0.0005;
setting.net.momentum                                = 0.9;
setting.net.modelType                               = 'dropout';
setting.net.learningRate                            = [ 0.01 * ones( 1, 8 ), 0.001 * ones( 1, 2 ) ];
setting.app.initDet.scaleStep                       = 2;
setting.app.initDet.numScale                        = 7;
setting.app.initDet.dvecLength                      = 30;
setting.app.initDet.numMaxTest                      = 50;
setting.app.initDet.docScaleMag                     = 4;
setting.app.initDet.startHorzScale                  = 1;
setting.app.initDet.horzScaleStep                   = 0.5;
setting.app.initDet.endHorzScale                    = 2;
setting.app.initDet.preBoundInitGuess               = false;
setting.app.initMrg.method                          = 'NMS';
setting.app.initMrg.overlap                         = 0.8;
setting.app.initMrg.minNumSuppBox                   = 1;
setting.app.initMrg.mergeType                       = 'WAVG';
setting.app.initMrg.scoreType                       = 'AVG';
setting.app.refine.dvecLength                       = 30;
setting.app.refine.boxScaleMag                      = 2.5;
setting.app.refine.method                           = 'OV';
setting.app.refine.overlap                          = 0.5;
setting.app.refine.minNumSuppBox                    = 0;
setting.app.refine.mergeType                        = 'WAVG';
setting.app.refine.scoreType                        = 'AVG';
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;
net = Net( io, setting.net );
net.init;
% Added.)
net.layers = net_.layers;
net.imStats.averageImage = net_.avgIm;
% End.)
app = DetSingleCls...
    ( db, net, ...
    setting.app.initDet, ...
    setting.app.initMrg, ...
    setting.app.refine );
app.init( setting.gpus );

%% DEV.
clc; clearvars -except db io net path setting app;
iids = 9941; db.getTeiids;
% app.settingInitMrg.overlap = 1;
% app.settingInitMrg.minNumSuppBox = 0;
% app.settingRefine.minNumSuppBox = 1;
app.patchSize = [ 227, 227 ];
app.settingInitMrg.overlap = 0.8;
app.settingInitMrg.minNumSuppBox = 1;
app.settingRefine.minNumSuppBox = 0;
for iid = iids',
    im = imread( db.iid2impath{ iid } );
    did2tlbr = app.iid2det0( iid );
    figure( 1 ); plottlbr( did2tlbr, im, false, { 'r', 'y', 'b', 'g' } );
    did2tlbr = app.iid2det( iid );
    figure( 2 ); plottlbr( did2tlbr, im, false, 'c' );
    if ~isempty( did2tlbr )
        did2tlbr = app.im2redet( im, did2tlbr );
    end
    figure( 3 ); plottlbr( did2tlbr, im, false, 'c' );
    if numel( iids ) ~= 1, waitforbuttonpress; end;
end;


%% PR.
% res = res1;
% fp = cumsum( res.rank2fp );
% tp = cumsum( res.rank2tp );
% npos = sum( db.oid2cid( unique( cat( 1, db.iid2oids{ db.iid2setid == 2 } ) ) ) == 15 & ( ~db.oid2diff( unique( cat( 1, db.iid2oids{ db.iid2setid == 2 } ) ) ) ) );
% rec = tp / npos;
% prec = tp ./ ( fp + tp );
% ap = 0;
% for t = 0 : 0.1 : 1
%     p = max( prec( rec >= t ) );
%     if isempty( p ), p = 0; end;
%     ap = ap + p / 11;
% end
% fprintf( 'AP: %.2f\n', ap * 100 );
% plot( rec, prec, '-r' );
% grid;
% xlim( [ 0, 1 ] );
% ylim( [ 0, 1 ] );
% xlabel( 'Recall' );
% ylabel( 'Precision' );
% hold on;


