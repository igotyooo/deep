%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 1;
setting.db                                          = path.db.ilsvrcdet2015; path.db.voc2007; 
setting.propObj.numScaling                          = 24; 12; 48; 
setting.propObj.dilate                              = 1 / 2;
setting.propObj.normalizeImageMaxSide               = 500;
setting.propObj.maximumImageSize                    = 9e6;
setting.propObj.posIntOverRegnMoreThan              = ( 1 / 2.4 ^ 2 ); ( 1 / 16 ); ( 1 / 8 ); ( 1 / 4 );
setting.propObj.directionVectorSize                 = 30;
setting.propObj.numTopClassification                = 3;        % 2 for PASCAL VOC.
setting.propObj.numTopDirection                     = 2;        % 2 for PASCAL VOC.
setting.det0.main.rescaleBox                        = 1;
setting.det0.main.directionVectorSize               = setting.propObj.directionVectorSize;
setting.det0.main.numTopClassification              = setting.propObj.numTopClassification;
setting.det0.main.numTopDirection                   = setting.propObj.numTopDirection;
setting.det0.main.numMaxTest                        = 50; 
setting.det0.post.mergingOverlap                    = 1; 0.85; 
setting.det0.post.mergingType                       = 'OV';
setting.det0.post.mergingMethod                     = 'WAVG';
setting.det0.post.minimumNumSupportBox              = 0; 1; 


%% DO THE JOB.
reset( gpuDevice( setting.gpus ) );
db = Db( setting.db, path.dstDir );
db.genDb;

%% DEV.
clc; clearvars -except db path setting;
caffe.reset_all(  );

rng( 'shuffle' );
iid = db.getTeiids;
% cid = 101; oids = find( db.iid2setid( db.oid2iid ) == 2 ); iid = unique( db.oid2iid( oids( db.oid2cid( oids ) == cid ) ) );
% 9370; 4330; 2072; 4134; 5291; 3220; 8623; 7903; 6982; 4219; 69; 558; 5126; 1267; 4288; 1844; 5174; 7769; 8593; 8578; 4; for VOC.
% 331716; 340953; 333993; 348692; 335237; 346222; 348717; 335179; 341630; 333968; 342155; for ILSVRC.
iid = randsample( iid', 1 ); 
db.demoBbox( 1, [ 3, 4, 1, 2 ], iid );

switch setting.db.name,
    case 'VOC2007',
        netInfo.modelPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07.caffemodel';
        netInfo.protoPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07.prototxt';
        netInfo.rgbMeanPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07_rgbmean.mat';
        netInfo.modelName = 'ANET_GOO_VOC07';
    case 'ILSVRCDET2015',
        netInfo.modelPath = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet_e6.caffemodel';
        netInfo.protoPath = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet.prototxt';
        netInfo.rgbMeanPath = '/iron/data/TRAINED_NETS_CAFFE/attNet_gnet_ilsdet_rgbmean.mat';
        netInfo.modelName = 'ANET_GOO_ILSDET07';
end;
netInfo.patchSide = 223;
netInfo.stride = 32;

prop = PropObjCaffe3( db, setting.propObj );
prop.init( netInfo, setting.gpus );
prop.demo( 2, [ 3, 4, 1, 3 ], false, iid );
caffe.reset_all(  );

netInfo.patchSide = 224; 
det0 = AttNetCaffe3( db, prop, setting.det0.main, setting.det0.post ); clear prop;
det0.init( netInfo, setting.gpus );
det0.demo2( 3, [ 3, 4, 1, 4 ], iid );
caffe.reset_all(  );








