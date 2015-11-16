%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all;
addpath( genpath( '../../' ) ); init_ilsvrc15;
setting.gpus                                = 1;
setting.db                                  = path.db.ilsvrcdet2015; 
setting.netInfo                             = path.attNetCaffe.ilsdet;
setting.attNetProp.flip                     = true; false; 
setting.attNetProp.normalizeImageMaxSide    = 500;
setting.attNetProp.numScaling               = 24; 
setting.attNetProp.dilate                   = 1 / 2;
setting.attNetProp.posIntOverRegnMoreThan   = 1 / 8; 
setting.attNetProp.maximumImageSize         = 9e6;
setting.attNetProp.numTopClassification     = 3;
setting.attNetProp.numTopDirection          = 2;
setting.attNetProp.directionVectorSize      = 30;
setting.attNetProp.minNumDetectionPerClass  = 0;
setting.attNetDet0.type                     = 'DYNAMIC';
setting.attNetDet0.rescaleBox               = 1;
setting.attNetDet0.numTopClassification     = setting.attNetProp.numTopClassification;
setting.attNetDet0.numTopDirection          = setting.attNetProp.numTopDirection;
setting.attNetDet0.directionVectorSize      = setting.attNetProp.directionVectorSize;
setting.attNetDet0.minNumDetectionPerClass  = 0;
setting.attNetMrg0.mergingOverlap           = 0.8; 
setting.attNetMrg0.mergingType              = 'NMS';
setting.attNetMrg0.mergingMethod            = 'MAX';
setting.attNetMrg0.minimumNumSupportBox     = 1; 
setting.attNetMrg0.classWiseMerging         = true;
setting.attNetDet1.type                     = 'STATIC';
setting.attNetDet1.rescaleBox               = 2.5; 
setting.attNetDet1.onlyTargetAndBackground  = true;
setting.attNetDet1.directionVectorSize      = 15;
setting.attNetDet1.minNumDetectionPerClass  = 1;
setting.attNetDet1.weightDirection          = 0;
setting.attNetMrg1.mergingOverlap           = 0.5;
setting.attNetMrg1.mergingType              = 'OV';
setting.attNetMrg1.mergingMethod            = 'MAX';
setting.attNetMrg1.minimumNumSupportBox     = 0;
setting.attNetMrg1.classWiseMerging         = true;
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;
attNet = AttNetCaffe( ...
    db, ...
    setting.attNetProp, ...
    setting.attNetDet0, ...
    setting.attNetMrg0, ...
    setting.attNetDet1, ...
    setting.attNetMrg1 );
attNet.init( setting.netInfo, setting.gpus );

%% DEMO.
clc; close all;
rng( 'shuffle' );
iid = db.getTeiids;
iid = randsample( iid', 1 ); 
db.demoBbox( 1, [ 3, 6, 1, 1 ], iid );
attNet.demoDet( iid, true );

%% CLEAN UP.
caffe.reset_all(  );



















