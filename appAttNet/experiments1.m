%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 2;
setting.db                                          = path.db.voc2007;
setting.io.tsDb.selectClassName                     = 'bottle';
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
setting.app.initDet.numScale                        = 7;
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



% 1. 
clearvars -except setting path;
settingNew = setting;
settingNew.io.tsDb.docScaleMag = 1;

reset( gpuDevice( settingNew.gpus ) ); 
db = Db( settingNew.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    settingNew.io.tsDb, ...
    settingNew.io.tsNet, ...
    settingNew.io.general );
io.init;
net = Net( io, settingNew.net );
net.init;
net.train( settingNew.gpus, 'visionresearchreport@gmail.com' );
net.fetchBestNet;
app = DetSingleCls...
    ( db, net, ...
    settingNew.app.initDet, ...
    settingNew.app.initMrg, ...
    settingNew.app.refine );
app.init( settingNew.gpus );
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

% 2.
clearvars -except setting path;
settingNew = setting;
settingNew.io.tsDb.docScaleMag = 4;

reset( gpuDevice( settingNew.gpus ) ); 
db = Db( settingNew.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    settingNew.io.tsDb, ...
    settingNew.io.tsNet, ...
    settingNew.io.general );
io.init;
net = Net( io, settingNew.net );
net.init;
net.train( settingNew.gpus, 'visionresearchreport@gmail.com' );
net.fetchBestNet;
app = DetSingleCls...
    ( db, net, ...
    settingNew.app.initDet, ...
    settingNew.app.initMrg, ...
    settingNew.app.refine );
app.init( settingNew.gpus );
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



% 3.
clearvars -except setting path;
settingNew = setting;
settingNew.io.tsDb.docScaleMag = 1;
settingNew.io.tsNet.pretrainedNetName = path.net.vgg_m.name;

reset( gpuDevice( settingNew.gpus ) ); 
db = Db( settingNew.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    settingNew.io.tsDb, ...
    settingNew.io.tsNet, ...
    settingNew.io.general );
io.init;
net = Net( io, settingNew.net );
net.init;
net.train( settingNew.gpus, 'visionresearchreport@gmail.com' );
net.fetchBestNet;
app = DetSingleCls...
    ( db, net, ...
    settingNew.app.initDet, ...
    settingNew.app.initMrg, ...
    settingNew.app.refine );
app.init( settingNew.gpus );
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

% 4.
clearvars -except setting path;
settingNew = setting;
settingNew.io.tsDb.docScaleMag = 4;
settingNew.io.tsNet.pretrainedNetName = path.net.vgg_m.name;

reset( gpuDevice( settingNew.gpus ) ); 
db = Db( settingNew.db, path.dstDir );
db.genDb;
io = InOutDetSingleCls( db, ...
    settingNew.io.tsDb, ...
    settingNew.io.tsNet, ...
    settingNew.io.general );
io.init;
net = Net( io, settingNew.net );
net.init;
net.train( settingNew.gpus, 'visionresearchreport@gmail.com' );
net.fetchBestNet;
app = DetSingleCls...
    ( db, net, ...
    settingNew.app.initDet, ...
    settingNew.app.initMrg, ...
    settingNew.app.refine );
app.init( settingNew.gpus );
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
