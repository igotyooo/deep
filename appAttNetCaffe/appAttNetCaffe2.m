%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all; 
addpath( genpath( '..' ) ); init;
setting.gpus                                        = 1;
setting.db                                          = path.db.voc2007; path.db.ilsvrcdet2015; 
setting.propObj.numScaling                          = 24; 48; 
setting.propObj.dilate                              = 1 / 2;
setting.propObj.normalizeImageMaxSide               = 500;
setting.propObj.posIntOverRegnMoreThan              = ( 1 / 8 ); ( 1 / 2.4 ^ 2 ); ( 1 / 4 ); 
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

%% DEV.
clc; clearvars -except db path setting;
caffe.reset_all(  );

rng( 'shuffle' );
iid = db.getTeiids;
iid = randsample( iid', 1 ); 5126; 1267; 4288; 1844; 5174; 7769; 4; 8593; 9370; 

netInfo.modelPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07.caffemodel';
netInfo.protoPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07.prototxt';
netInfo.rgbMeanPath = '/iron/data/TRAINED_NETS_CAFFE/attnet_gnet_voc07_rgbmean.mat';
netInfo.modelName = 'ANET_GOO_VOC07';
netInfo.patchSide = 223; 
netInfo.stride = 32;

prop = PropObjCaffe( db, setting.propObj );
prop.init( netInfo, setting.gpus );
prop.demo( 1, false, iid );
caffe.reset_all(  );

det0 = AttNetCaffe( db, prop, setting.det0.main, setting.det0.post ); clear prop;
det0.init( netInfo, setting.gpus );
det0.demo2( 2, iid );
caffe.reset_all(  );