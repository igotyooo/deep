%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 1;
setting.db                                          = path.db.voc2007;
setting.io.tsDb.selectClassName                     = 'dog';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.dstSide                             = 227;
setting.io.tsDb.numScale                            = 10;
setting.io.tsDb.scaleStep                           = 2;
setting.io.tsDb.numAspect                           = 16;
setting.io.tsDb.docScaleMag                         = 4;
setting.io.tsDb.confidence                          = 0.97;
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
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m_2048.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 10;
setting.io.tsNet.outFilterDepth                     = 2048;
setting.io.general.dstSide                          = 227;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128 * numel( setting.gpus );
setting.net.normalizeImage                          = 'NONE';
setting.net.weightDecay                             = 0.0005;
setting.net.momentum                                = 0.9;
setting.net.modelType                               = 'dropout';
setting.net.learningRate                            = [ 0.01 * ones( 1, 8 ), 0.001 * ones( 1, 2 ) ];
setting.app.initDet.scaleStep                       = 2;
setting.app.initDet.numScale                        = 6; % 7;
setting.app.initDet.dvecLength                      = 30;
setting.app.initDet.numMaxTest                      = 50;
setting.app.initDet.patchMargin                     = 0.5;
setting.app.initDet.numAspect                       = 16 / 2;
setting.app.initDet.confidence                      = 0.97;
setting.app.initMrg.selectScaleIds                  = 1 : setting.app.initDet.numScale;
setting.app.initMrg.selectAspectIds                 = 1 : setting.app.initDet.numAspect;
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

reset( gpuDevice( setting.gpus ) ); 
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;
net = Net( io, setting.net );
net.init;
net.train( setting.gpus, 'visionresearchreport@gmail.com' );
net.fetchBestNet;
app = DetSingleCls...
    ( db, net, ...
    setting.app.initDet, ...
    setting.app.initMrg, ...
    setting.app.refine );
app.init( setting.gpus );
app.detDb;
[   res0.ap, ...
    res0.rank2iid, ...
    res0.rank2bbox, ...
    res0.rank2tp, ...
    res0.rank2fp ] = ...
    app.computeAp...
    ( 'visionresearchreport@gmail.com' );
app.refineDet( 1 );
[   res1.ap, ...
    res1.rank2iid, ...
    res1.rank2bbox, ...
    res1.rank2tp, ...
    res1.rank2fp ] = ...
    app.computeAp...
    ( 'visionresearchreport@gmail.com' );





%% DEV.
% clc; clearvars -except db io net path setting app;
% % cid = find( cellfun( @( name )strcmp( name, io.settingTsDb.selectClassName ), db.cid2name ) );
% % iids = setdiff( unique( db.oid2iid( db.oid2cid == cid ) ), find( db.iid2setid == 1 ) );
% iids = 2614; db.getTeiids;
% for iid = iids',
%     im = imread( db.iid2impath{ iid } );
%     did2tlbr = app.iid2det0( iid );
%     figure( 1 ); plottlbr( did2tlbr, im, false, { 'r', 'y', 'b', 'g' } );
%     did2tlbr = app.iid2det( iid );
%     figure( 2 ); plottlbr( did2tlbr, im, false, 'c' );
%     did2str = {  };
%     if ~isempty( did2tlbr )
%         [ did2tlbr, did2rescore ]= app.im2redet( im, did2tlbr );
%         did2str = cellfun( @num2str, num2cell( did2rescore ), 'uniformOutput', false );
%     end
%     figure( 3 ); plottlbr( did2tlbr, im, false, 'c', did2str );
%     title( num2str( iid ) );
%     if numel( iids ) ~= 1, waitforbuttonpress; end;
% end;


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
