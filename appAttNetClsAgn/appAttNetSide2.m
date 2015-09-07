%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 1;
setting.db                                          = path.db.coco2014; path.db.voc2007; 
setting.io.tsDb.numScaling                          = 256;
setting.io.tsDb.dilate                              = 1 / 4;
setting.io.tsDb.directionVectorMagnitude            = 10;
setting.io.tsDb.numMaxRegionPerDirectionPair        = 16;
setting.io.tsDb.posMinMargin                        = 0; 0.1;
setting.io.tsDb.posIntOverRegnMoreThan              = 1 / 4;
setting.io.tsDb.posIntOverTarObjMoreThan            = 0.99;
setting.io.tsDb.posIntOverSubObjMoreThan            = 0.7;
setting.io.tsDb.posIntMajorityMoreThan              = 2;
setting.io.tsDb.snegIntOverRegnMoreThan             = 1 / 36;
setting.io.tsDb.snegIntOverObjMoreThan              = 0.3;
setting.io.tsDb.snegIntOverObjLessThan              = 0.7;
setting.io.tsDb.negIntOverObjLessThan               = 0.1;
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 4;
setting.io.general.shuffleSequance                  = false;
setting.io.general.batchSize                        = 128;
setting.net.normalizeImage                          = 'NONE';
setting.net.weightDecay                             = 0.0005;
setting.net.momentum                                = 0.9;
setting.net.modelType                               = 'dropout';
setting.net.learningRate                            = [ 0.01 * ones( 1, 17 ), 0.001 * ones( 1, 10 ) ];
setting.propObj.main.numScaling                     = 24;
setting.propObj.main.dilate                         = setting.io.tsDb.dilate;
setting.propObj.main.posIntOverRegnMoreThan         = setting.io.tsDb.posIntOverRegnMoreThan;
setting.propObj.post.overlap                        = 0.7;

%% DO THE JOB.
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutAttNetSide2( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;
io.makeTsDb;
net = Net( io, setting.net );
net.init;
net.train( setting.gpus, ...
    'visionresearchreport@gmail.com' );
% net.fetchBestNet;
[ net, netName ] = net.provdNet;
net.name = netName;
net.normalization.averageImage = io.rgbMean;
propObj = PropObj( db, net, ...
    setting.propObj.main, setting.propObj.post );
propObj.init( setting.gpus );
propObj.propObj;




%% TEST PROPOSAL.
% clc; close all; clearvars -except db io path setting propObj net netName;
% iid = db.getTeiids;
% iid = randsample( iid', 1 );
% im = imread( db.iid2impath{ iid } );
% [ rid2tlbr, rid2score, rid2cid ] = propObj.iid2prop( iid );
% figure( 1 ); plottlbr( db.oid2bbox( :, db.iid2oids{ iid } ), db.iid2impath{ iid }, false, 'r' );
% figure( 2 ); 
% for rid = 1 : numel( rid2score ),
%     plottlbr( rid2tlbr( :, rid ), im, false, 'r', ...
%         { sprintf( '%s, %.2f', db.cid2name{ rid2cid( rid ) }, rid2score( rid ) ) } );
%     waitforbuttonpress;
% end;


%% TEST I/O.
% clc; close all; clearvars -except db io path setting;
% setid = 2;
% if setid == 1, 
%     [ ims, gts, iids ] = io.provdBchTr; 
%     tsdb = io.tsDb.tr;
% else
%     [ ims, gts, iids ] = io.provdBchVal; 
%     tsdb = io.tsDb.val;
% end;
% for s = 1 : setting.io.general.batchSize,
%     im = ims( :, :, :, s );
%     gt = gts( :, :, :, s );
%     gt = gt( : );
%     im = uint8( bsxfun( @plus, im, io.rgbMean ) );
%     try cname = db.cid2name{ gt( 1 ) }; catch, cname = 'bgd'; end;
%     try direction = mat2str( io.directions.dpid2dp( :, gt( 2 ) )' ); catch, direction = 'bgd'; end;
%     figure( 1 ); 
%     [ ~, iid ] = fileparts( tsdb.iid2impath{ iids( s ) } );
%     iid = str2double( iid ); plottlbr( db.oid2bbox( :, db.iid2oids{ iid } ), db.iid2impath{ iid }, false, 'r' );
%     figure( 2 ); 
%     imshow( im ); title( sprintf( '%s, %s (%d/%d)', cname, direction, s, setting.io.general.batchSize ) );
%     waitforbuttonpress;
% end;





