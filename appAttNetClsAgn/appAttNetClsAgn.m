%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 2;
setting.db                                          = path.db.voc2007;
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.patchSide                           = 227;
setting.io.tsDb.numScale                            = 16;
setting.io.tsDb.numAspect                           = 16;
setting.io.tsDb.confidence                          = 0.97;
setting.io.tsDb.dilate                              = 1 / 4;
setting.io.tsDb.posMinMargin                        = 0.1;
setting.io.tsDb.posIntOverRegnMoreThan              = 1 / 3;
setting.io.tsDb.posIntOverTarObjMoreThan            = 0.99;
setting.io.tsDb.posIntOverSubObjMoreThan            = 0.6;
setting.io.tsDb.posIntMajorityMoreThan              = 2;
setting.io.tsDb.snegIntOverRegnMoreThan             = 1 / 36;
setting.io.tsDb.snegIntOverObjMoreThan              = 0.3;
setting.io.tsDb.snegIntOverObjLessThan              = 0.7;
setting.io.tsDb.negIntOverObjLessThan               = 0.1;
setting.io.tsNet.pretrainedNetName                  = path.net.vgg_m.name;
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 4; % 1 / 10;
setting.io.general.shuffleSequance                  = false;
setting.io.general.dstSide                          = setting.io.tsDb.patchSide;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128;
setting.net.normalizeImage                          = 'NONE';
setting.net.weightDecay                             = 0.0005;
setting.net.momentum                                = 0.9;
setting.net.modelType                               = 'dropout';
setting.net.learningRate                            = [ 0.01 * ones( 1, 17 ), 0.001 * ones( 1, 10 ) ];
setting.propObj.numScale                            = 6;
setting.propObj.numAspect                           = 6;
setting.propObj.confidence                          = 0.90;
setting.propObj.dilate                              = setting.io.tsDb.dilate;
setting.propObj.posIntOverRegnMoreThan              = setting.io.tsDb.posIntOverRegnMoreThan;

%% DO THE JOB.
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutPropRegn( db, ...
    setting.io.tsDb, ...
    setting.io.tsNet, ...
    setting.io.general );
io.init;
net = Net( io, setting.net );
net.init;
net.train( setting.gpus, ...
    'visionresearchreport@gmail.com' );
net.fetchBestNet;
[ net, netName ] = net.provdNet;
net.layers{ end }.type = 'softmax';
net.normalization.averageImage = io.rgbMean;
propObj = PropObj( db, net, setting.propObj );
propObj.init( setting.gpus );







%% 
% clc;
% [ ims, gts ] = io.provdBchTr;
% for s = 1 : setting.io.general.batchSize,
%     im = ims( :, :, :, s );
%     gt = gts( s );
%     im = uint8( bsxfun( @plus, im, io.rgbMean ) );
%     try cname = db.cid2name{ gt }; catch, cname = 'background'; end;
%     imshow( im ); title( sprintf( '%s (%d/%d)', cname, s, setting.io.general.batchSize ) );
%     waitforbuttonpress;
% end;




%% 
clc; close all; clearvars -except db io net path setting propObj;
iid = randsample( db.getTeiids, 1 );
im = imread( db.iid2impath{ iid } );
sid2npix = round( 227 * 2 .^ ( 0 : 0.5 : 2.5 ) );
sid2size = scaleImage( sid2npix, 'MIN', size( im ) );
[ rid2tlbr, rid2desc ] = extActivationMap( im, propObj.propNet, 21, sid2size, 'bicubic', 219, 32 );

[ rank2score, rank2cid ] = sort( rid2desc, 'descend' );
rid2score = rank2score( 1, : ) - sum( rank2score( 2 : end, : ), 1 );
rid2cid = rank2cid( 1, : );
rid2isfgd = rid2cid ~= 21;

frid2score = rid2score( rid2isfgd );
frid2tlbr = rid2tlbr( :, rid2isfgd );
frid2cid = rid2cid( rid2isfgd );
[ rank2score, rank2frid ] = sort( frid2score, 'descend' );
rank2tlbr = frid2tlbr( :, rank2frid );
rank2cid = frid2cid( rank2frid );
rank2cname = db.cid2name( rank2cid );

plottlbr( rank2tlbr, im, true, 'r', cellfun( @( cname, score )strcat( cname, ':', num2str( score ) ), rank2cname', num2cell( rank2score ), 'UniformOutput', false ) );


















