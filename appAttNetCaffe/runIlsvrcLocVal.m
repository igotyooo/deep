function runIlsvrcLocVal( numDiv, divId, gpuId )
    clc; clearvars -except numDiv divId gpuId; fclose all; close all;
    addpath( genpath( '..' ) ); init_ilsvrc15;
    setting.gpus                                = gpuId;
    setting.db                                  = path.db.ilsvrcclsloc2015;
    setting.netInfo                             = path.attNetCaffe.ilsloc;
    setting.attNetProp.normalizeImageMaxSide    = 500;
    setting.attNetProp.numScaling               = 6;
    setting.attNetProp.dilate                   = 1 / 2;
    setting.attNetProp.posIntOverRegnMoreThan   = 1 / 8;
    setting.attNetProp.maximumImageSize         = 9e6;
    setting.attNetProp.numTopClassification     = 10;
    setting.attNetProp.numTopDirection          = 1;
    setting.attNetProp.onlyTargetAndBackground  = false;
    setting.attNetProp.directionVectorSize      = 30;
    setting.attNetProp.minNumDetectionPerClass  = 3;
    setting.attNetDet0.type                     = 'STATIC';
    setting.attNetDet0.rescaleBox               = 1;
    setting.attNetDet0.onlyTargetAndBackground  = setting.attNetProp.onlyTargetAndBackground;
    setting.attNetDet0.directionVectorSize      = setting.attNetProp.directionVectorSize;
    setting.attNetDet0.minNumDetectionPerClass  = setting.attNetProp.minNumDetectionPerClass;
    reset( gpuDevice( setting.gpus ) );
    db = Db( setting.db, path.dstDir );
    db.genDb;
    attNet = AttNetCaffe( db, setting.attNetProp, setting.attNetDet0, [  ], [  ], [  ] );
    attNet.init( setting.netInfo, setting.gpus );
    attNet.subDbDet0( numDiv, divId );
    caffe.reset_all(  );
end