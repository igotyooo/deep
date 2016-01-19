%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 1;
setting.db                                          = path.db.voc2007; path.db.ilsvrcdet2015; path.db.coco2014; 
setting.io.tsDb.numScaling                          = 24;
setting.io.tsDb.dilate                              = 1 / 4;
setting.io.tsDb.normalizeImageMaxSide               = 500;
setting.io.tsDb.posGotoMargin                       = 2.4;
setting.io.tsDb.numQuantizeBetweenStopAndGoto       = 3;
setting.io.tsDb.negIntOverObjLessThan               = 0.1;
setting.io.net.pretrainedNetName                    = path.net.vgg_m.name;
setting.io.net.suppressPretrainedLayerLearnRate     = 1 / 4;
setting.io.general.shuffleSequance                  = false; 
setting.io.general.batchSize                        = 128;
setting.io.general.numGoSmaplePerObj                = 1;
setting.io.general.numAnyDirectionSmaplePerObj      = 14; 2; 
setting.io.general.numStopSmaplePerObj              = 1;
setting.io.general.numBackgroundSmaplePerObj        = 16; 4; 
setting.net.normalizeImage                          = 'NONE';
setting.net.weightDecay                             = 0.0005;
setting.net.momentum                                = 0.9;
setting.net.modelType                               = 'dropout';
setting.net.learningRate                            = [ 0.01 * ones( 1, 8 ), 0.001 * ones( 1, 2 ) ];
setting.propObj.numScaling                          = 24; 48; 
setting.propObj.dilate                              = 1 / 2;
setting.propObj.normalizeImageMaxSide               = setting.io.tsDb.normalizeImageMaxSide;
setting.propObj.posIntOverRegnMoreThan              = 1 / 8; 1 / setting.io.tsDb.posGotoMargin ^ 2; 1 / 4; 
setting.det0.main.rescaleBox                        = 1;
setting.det0.main.directionVectorSize               = 30; 15; 
setting.det0.main.numMaxTest                        = 50; 
setting.det0.post.mergingOverlap                    = 1; 0.85; 
setting.det0.post.mergingType                       = 'OV';
setting.det0.post.mergingMethod                     = 'WAVG';
setting.det0.post.minimumNumSupportBox              = 0; 1; 


%% DO THE JOB.
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
io = InOutAttNetCornerPerCls2( db, ...
    setting.io.tsDb, ...
    setting.io.net, ...
    setting.io.general );
io.init;
% io.makeTsDb;
net = Net( io, setting.net );
net.init;
net.train( setting.gpus, ...
    'visionresearchreport@gmail.com' );
net.fetchBestNet;
[ net, netName ] = net.provdNet;
net.name = netName;
net.normalization.averageImage = io.rgbMean;
propObj = PropObjCornerPerCls2...
    ( db, net, setting.propObj );
propObj.init( setting.gpus );
det0 = AttNetCornerPerCls2( db, net, propObj, ...
    setting.det0.main, setting.det0.post );
det0.init( setting.gpus );


%% DEMO.
clear propObj det0 det1;
rng( 'shuffle' );
iid = db.getTeiids;
iid = 2758; 1267; 9370; 8578; 5174; 7769; 69; 7617; 8363; randsample( iid', 1 ); 1363; 3096; 7500; 178; 3187; 4134; 
propObj = PropObjCornerPerCls2...
    ( db, net, setting.propObj );
propObj.init( setting.gpus );
propObj.demo( 1, false, iid );

det0 = AttNetCornerPerCls2( db, net, propObj, ...
    setting.det0.main, setting.det0.post ); clear propObj;
det0.init( setting.gpus );
det0.demo2( 2, iid );









