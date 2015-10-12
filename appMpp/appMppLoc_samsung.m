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
setting.imDscrber.weights               = 1;
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
imDscrber = ImDscrber( db, { fisher; }, setting.imDscrber );
imDscrber.descDb;
svm = Svm( db, imDscrber, setting.svm );
svm.trainSvm;
svm.evalSvm( 'visionresearchreport@gmail.com' );


%% DEMO: LOCALIZATION.
clc; clearvars -except db fisher imDscrber net neuralDesc neuralRegnDscrber path setting svm map;
% Set parameters.
numMaxVote = 150; % 300; for light-on.
mapsize = 256;
scoreThrsh = -Inf;
smoothMap = round( mapsize / 20 );
% Select a test image.
targetCid = 7;
iid = db.oid2iid( db.oid2cid == targetCid );
iid = iid( db.iid2setid( iid ) == 2 );
iid = randsample( iid', 1 ); 54; 22; 
im = imread( db.iid2impath{ iid } );
% Prepare for data.
means = neuralRegnDscrber.gmm.means;
covs = neuralRegnDscrber.gmm.covs;
priors = neuralRegnDscrber.gmm.priors;
cid2w = svm.loadSvm;
% Extract neural activtions.
fprintf( 'Ext regn descs.\n' );
[ rid2tlbr, rid2desc ] = ...
    neuralRegnDscrber.iid2regdesc( iid, false );
[ r, c, ~ ] = size( im );
rid2tlbr = resizeTlbr( rid2tlbr, [ r; c; ], [ mapsize; mapsize; ] );
rid2tlbr = round( rid2tlbr );
fprintf( 'Done.\n' );
% Encode fisher.
fprintf( 'Encode per-desc Fisher.\n' );
fisherDim = 2 * size( rid2desc, 1 ) * size( means, 2 );
numRegn = size( rid2tlbr, 2 );
rid2fisher = vl_fisher...
    ( rid2desc, means, covs, priors, 'NoAveragePooling' );
rid2fisher = reshape( rid2fisher, [ fisherDim, numRegn ] );
rid2fisher = kernelMap( rid2fisher, 'HELL' );
rid2fisher = nmlzVecs( rid2fisher, 'L2' );
fprintf( 'Done.\n' );
% Make a score map.
fprintf( 'Compute score map.\n' );
cid2rid2s = cid2w' * cat( 1, rid2fisher, ones( 1, numRegn ) );
[ rank2rid2s, rank2rid2cid ] = sort( cid2rid2s, 1, 'descend' );
[ ~, ridBest ] = max( cid2rid2s( targetCid, : ) );
rids = rank2rid2cid( 1, : ) == targetCid & rank2rid2s( 1, : ) > scoreThrsh;
rids = find( rids );
scoremap = zeros( mapsize, 'single' );
for rid = rids,
    r1 = rid2tlbr( 1, rid ); c1 = rid2tlbr( 2, rid );
    r2 = rid2tlbr( 3, rid ); c2 = rid2tlbr( 4, rid );
    scoremap( r1 : r2, c1 : c2 ) = scoremap( r1 : r2, c1 : c2 ) + 1;
end;
scoremap = vl_imsmooth( scoremap, smoothMap );
[ ~, idx ] = max( scoremap( : ) );
bestc = ceil( idx / mapsize );
bestr = idx - mapsize * ( bestc - 1 );
fprintf( 'Done.\n' );
% Display.
scoremap = uint8( scoremap * 256 / numMaxVote );
scoremap = uint8( ind2rgb( gray2ind( scoremap ), colormap ) * 255 );

bestRegn = rid2tlbr( :, ridBest );
im = imresize( im, [ mapsize, mapsize ] );
imRes = cat( 2, im, scoremap );
cname = db.cid2name{ targetCid };
cname( cname == '_' ) = ' ';
imshow( imRes ); hold on;
plot( bestc, bestr, 'r+', 'MarkerSize', 52, 'LineWidth', 3 ); hold off;
set( gcf, 'Color', 'w' );
title( cname );
% plottlbr( bestRegn, imRes, false, 'r', { cname } );







