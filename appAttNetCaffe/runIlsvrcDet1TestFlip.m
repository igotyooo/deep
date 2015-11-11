function runIlsvrcDet1TestFlip( numDiv, divId, gpuId )
    clc; clearvars -except numDiv divId gpuId; fclose all; close all;
    addpath( genpath( '..' ) ); init_ilsvrc15;
    setting.gpus                                = gpuId;
    setting.db                                  = path.db.ilsvrcdet2015;
    setting.dbte                                = path.db.ilsvrcdet2015te;
    setting.netInfo                             = path.attNetCaffe.ilsdet;
    setting.attNetProp.flip                     = true; 
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
    setting.attNetMrg1.mergingOverlap           = 0.5;
    setting.attNetMrg1.mergingType              = 'OV';
    setting.attNetMrg1.mergingMethod            = 'MAX';
    setting.attNetMrg1.minimumNumSupportBox     = 0;
    setting.attNetMrg1.classWiseMerging         = true;
    reset( gpuDevice( setting.gpus ) );
    db = Db( setting.db, path.dstDir );
    db.genDb;
    dbte = Db( setting.dbte, path.dstDir );
    dbte.genDb;
    attNet = AttNetCaffe( db, setting.attNetProp, setting.attNetDet0, setting.attNetMrg0, setting.attNetDet1, setting.attNetMrg1 );
    attNet.init( setting.netInfo, setting.gpus );
    attNet.db = dbte;
    attNet.subDbDet0( numDiv, divId );
    attNet.subDbDet1( numDiv, divId );
    caffe.reset_all(  );
end