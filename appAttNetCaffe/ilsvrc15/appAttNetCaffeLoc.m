%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all;
addpath( genpath( '../../' ) ); init_ilsvrc15;
setting.gpus                                = 2;
setting.db                                  = path.db.ilsvrcclsloc2015;
setting.netInfo                             = path.attNetCaffe.ilsloc;
setting.attNetProp.flip                     = false;
setting.attNetProp.normalizeImageMaxSide    = 500;
setting.attNetProp.numScaling               = 6;
setting.attNetProp.dilate                   = 1 / 2;        
setting.attNetProp.posIntOverRegnMoreThan   = 1 / 16;   % 1 / 8;
setting.attNetProp.maximumImageSize         = 9e6;
setting.attNetProp.numTopClassification     = 5;        % 10;
setting.attNetProp.numTopDirection          = 1; 
setting.attNetProp.onlyTargetAndBackground  = true;     % false;
setting.attNetProp.directionVectorSize      = 30;
setting.attNetProp.minNumDetectionPerClass  = 10;       % 3;
setting.attNetDet0.type                     = 'STATIC';
setting.attNetDet0.rescaleBox               = 1;
setting.attNetDet0.onlyTargetAndBackground  = setting.attNetProp.onlyTargetAndBackground;
setting.attNetDet0.directionVectorSize      = setting.attNetProp.directionVectorSize;
setting.attNetDet0.minNumDetectionPerClass  = setting.attNetProp.minNumDetectionPerClass; 
setting.attNetMrg0.mergingOverlap           = 1;
setting.attNetMrg0.mergingType              = 'OV';
setting.attNetMrg0.mergingMethod            = 'MAX';
setting.attNetMrg0.minimumNumSupportBox     = 0;

setting.attNetDet1.type                     = 'STATIC';
setting.attNetDet1.rescaleBox               = 8; 
setting.attNetDet1.onlyTargetAndBackground  = setting.attNetProp.onlyTargetAndBackground;
setting.attNetDet1.directionVectorSize      = 15;
setting.attNetDet1.minNumDetectionPerClass  = 1;
setting.attNetMrg1.mergingOverlap           = 0.5;
setting.attNetMrg1.mergingType              = 'OV';
setting.attNetMrg1.mergingMethod            = 'MAX';
setting.attNetMrg1.minimumNumSupportBox     = 0;

reset( gpuDevice( setting.gpus ) );
caffe.reset_all(  );
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

%% PROCESS DB.
attNet.subDbDet0( 1, 1 );
attNet.subDbDet1( 1, 1 );

%% DEMO.
clc; close all;
rng( 'shuffle' );
iid = db.getTeiids;
iid = randsample( iid', 1 ); 544875; 561994; 592142; 545648; 581597; 564875; 574265; 561999; 563081; 570161; 559398; 568965; 579394; 
% 581518; 550070; 592378; 571030; Debug these ids!!!!!
db.demoBbox( 1, [ 3, 6, 1, 1 ], iid );
attNet.demoDet( iid, true );