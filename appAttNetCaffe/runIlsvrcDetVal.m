function runIlsvrcDetVal( numDiv, divId, gpuId )
    try
        clc; clearvars -except gpuid numGpu; fclose all; close all;
        addpath( genpath( '..' ) ); init;
        setting.gpus                                = gpuId;
        setting.db                                  = path.db.ilsvrcdet2015;
        setting.netInfo                             = path.attNetCaffe.ilsdet;
        setting.attNetProp.numBaseProposal          = 0;
        setting.attNetProp.normalizeImageMaxSide    = 500;
        setting.attNetProp.numScaling               = 24; 
        setting.attNetProp.dilate                   = 1 / 2;
        setting.attNetProp.posIntOverRegnMoreThan   = ( 1 / 8 ); 
        setting.attNetProp.maximumImageSize         = 9e6;
        setting.attNetProp.numTopClassification     = 3;
        setting.attNetProp.numTopDirection          = 2;
        setting.attNetProp.directionVectorSize      = 30;
        setting.attNetDet0.type                     = 'DYNAMIC';
        setting.attNetDet0.rescaleBox               = 1;
        setting.attNetDet0.numTopClassification     = setting.attNetProp.numTopClassification;
        setting.attNetDet0.numTopDirection          = setting.attNetProp.numTopDirection;
        setting.attNetDet0.directionVectorSize      = setting.attNetProp.directionVectorSize;
        reset( gpuDevice( setting.gpus ) );
        db = Db( setting.db, path.dstDir );
        db.genDb;
        cidx2cid = 1 : db.getNumClass;
        attNet = AttNetCaffe( db, setting.attNetProp, setting.attNetDet0, [  ], [  ], [  ] );
        attNet.init( setting.netInfo, setting.gpus );
        attNet.subDbDet0( cidx2cid, numDiv, divId );
        caffe.reset_all(  );
    catch ME,
        ME.stack
        [ ~, hostName ] = system( 'hostname' );
        title = sprintf( 'PROCESSING KILLED! HOST: %s, GPU_ID: %d.', upper( hostName( 1 : end - 1 ) ), gpuid );
        mssg{ 1 } = [ 'Message: ', ME.message ];
        mssg{ 2 } = [ 'File: ', ME.stack.file ];
        mssg{ 3 } = [ 'Name: ', ME.stack.name ];
        mssg{ 4 } = [ 'Line: ', num2str( ME.stack.line ) ];
        fprintf( '%s\n', mssg{ : } );
        sendEmail( 'visionresearchreport@gmail.com', 'visionresearchreporter', 'overmars29@gmail.com', title, mssg, '' );
    end;
end