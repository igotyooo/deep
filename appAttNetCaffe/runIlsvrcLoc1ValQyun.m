function runIlsvrcLoc1ValQyun( numDiv, divId, gpuId )
    clc; clearvars -except numDiv divId gpuId; fclose all; close all;
    addpath( genpath( '..' ) ); init_ilsvrc15;
    setting.gpus                                = gpuId;
    setting.db                                  = path.db.ilsvrcclsloc2015;
    setting.netInfo                             = path.attNetCaffe.ilsloc;
    setting.attNetMrg0.mergingOverlap           = 1;
    setting.attNetMrg0.mergingType              = 'OV';
    setting.attNetMrg0.mergingMethod            = 'MAX';
    setting.attNetMrg0.minimumNumSupportBox     = 0;
    setting.attNetDet1.type                     = 'STATIC';
    setting.attNetDet1.rescaleBox               = 8;
    setting.attNetDet1.onlyTargetAndBackground  = setting.attNetProp.onlyTargetAndBackground;
    setting.attNetDet1.directionVectorSize      = 15;
    setting.attNetDet1.minNumDetectionPerClass  = 1;
    reset( gpuDevice( setting.gpus ) );
    db = Db( setting.db, path.dstDir );
    db.genDb;
    attNet = AttNetCaffe( db, [  ], [  ], setting.attNetMrg0, setting.attNetDet1, [  ] );
    attNet.init( setting.netInfo, setting.gpus );
    attNet.subDbDet1( numDiv, divId );
    caffe.reset_all(  );
end