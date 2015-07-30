%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 1;
setting.db                                          = path.db.voc2007;
setting.io.tsDb.selectClassName                     = 'person';
setting.io.tsDb.stride                              = 32;
setting.io.tsDb.patchSide                           = 227;
setting.io.tsDb.numScale                            = 16;
setting.io.tsDb.numAspect                           = 16;
setting.io.tsDb.confidence                          = 0.97;
setting.io.tsDb.violate                             = 1 / 4;
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
setting.io.tsNet.suppressPretrainedLayerLearnRate   = 1 / 10;
setting.io.general.dstSide                          = setting.io.tsDb.patchSide;
setting.io.general.dstCh                            = 3;
setting.io.general.batchSize                        = 128;
setting.net.normalizeImage                          = 'NONE';
setting.net.weightDecay                             = 0.0005;
setting.net.momentum                                = 0.9;
setting.net.modelType                               = 'dropout';
setting.net.learningRate                            = [ 0.01 * ones( 1, 80 ), 0.001 * ones( 1, 20 ) ];
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
net.train( setting.gpus, 'visionresearchreport@gmail.com' );
net.fetchBestNet;








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























