function runIlsvrcDetVal( numDiv, divId, gpuId )
    clc; clearvars -except numDiv divId gpuId; fclose all; close all;
    addpath( genpath( '..' ) ); init_ilsvrc15;
    setting.gpus                                = gpuId;
    setting.db                                  = path.db.ilsvrcdet2015;
    setting.netInfo                             = path.attNetCaffe.ilsdet;
    setting.attNetProp.normalizeImageMaxSide    = 500;
    setting.attNetProp.numScaling               = 24;
    setting.attNetProp.dilate                   = 1 / 2;
    setting.attNetProp.posIntOverRegnMoreThan   = ( 1 / 8 );
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
    reset( gpuDevice( setting.gpus ) );
    db = Db( setting.db, path.dstDir );
    db.genDb;
    attNet = AttNetCaffe( db, setting.attNetProp, setting.attNetDet0, [  ], [  ], [  ] );
    attNet.init( setting.netInfo, setting.gpus );
    attNet.subDbDet0( numDiv, divId );
    caffe.reset_all(  );
end