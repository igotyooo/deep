%% SET PARAMETERS ONLY.
clc; close all; fclose all; clear all;
addpath( genpath( '..' ) ); init; 
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

%% ILSVRCDET2015 EVALUATION.
numDiv = 20; divId = 1;
metaPath = fullfile( path.lib.ilsvrcDevKit, '/data/meta_det.mat' );
balckPath = fullfile( path.lib.ilsvrcDevKit, '/data/ILSVRC2015_det_validation_blacklist.txt' );
gtPath = fullfile( path.lib.ilsvrcDevKit, '/data/ILSVRC2015_det_validation_ground_truth.mat' );
[ det1.rid2tlbr, det1.rid2score, det1.rid2cid, det1.rid2iid, det1.idx2iid ] = ...
    attNet.getSubDbDet1( numDiv, divId );
rid2tlbr = det1.rid2tlbr;
if setting.attNetProp.flip,
    fprintf( 'Flip boxes.\n' );
    for iid = det1.idx2iid',
        rid2ok = det1.rid2iid == iid;
        tlbrs = rid2tlbr( :, rid2ok );
        width = db.iid2size( 2, iid );
        rid2tlbr( :, rid2ok ) = flipTlbr( tlbrs, width );
    end;
    fprintf( 'Done.\n' );
end;
det1.rid2iid0 = det1.rid2iid - db.getNumTrIm;
det1.idx2iid0 = det1.idx2iid - db.getNumTrIm;
det1.rid2pred = [ det1.rid2iid0, det1.rid2cid, det1.rid2score, rid2tlbr( [ 2; 1; 4; 3; ], : )' ];
idx2oids = db.iid2oids( det1.idx2iid );
objCids =db.oid2cid( cat( 1, idx2oids{ : } ) );
det1.cid2numObj = arrayfun( @( cid )sum( objCids == cid ), 1 : db.getNumClass )';
fprintf( 'Evaluation.\n' );
[ det1.ap, det1.recall, det1.prec ] = my_eval_ilsvrc_det...
    ( det1.idx2iid0, det1.rid2pred, det1.cid2numObj, metaPath, balckPath, gtPath );
mssg{ 1 } = sprintf( 'AP1: %.2f%%', mean( det1.ap( ~isnan( det1.ap ) ) ) * 100 );
cellfun( @disp, mssg' ); pause( 5 );

%% SAVE
det1 = rmfield( det1, 'rid2pred' );
det1 = rmfield( det1, 'cid2numObj' );
det1 = rmfield( det1, 'ap' );
det1 = rmfield( det1, 'recall' );
det1 = rmfield( det1, 'prec' );
fname = 'ANET_DETRES_MO0P5';
if setting.attNetProp.flip, fname = strcat( fname, '_FLIP' ); end;
fpath = fullfile( db.dstDir, strcat( fname, '.mat' ) );
save( fpath, 'det1' );


%% 
clc; close all;
rng( 'shuffle' );
iid = randsample( det1.idx2iid', 1 );
db.demoBbox( 1, [ 3, 6, 1, 1 ], iid );
attNet.demoDet( iid, true );


%% DEMO.
clc; close all;
rng( 'shuffle' );
iid = db.getTeiids;
% cid = 102; oids = find( db.iid2setid( db.oid2iid ) == 2 ); iid = unique( db.oid2iid( oids( db.oid2cid( oids ) == cid ) ) );
% 5126; 1267; 4288; 1844; 5174; 7769; 8593; 8578; 558; 69; 4219; 6982; 7903; 8623; 3220; 5291; 4134; 2072; 4; 4330; 9370; for VOC.
% 348717; 341630; 333993; 340953; 346222; 335179; 342155; 348692; 335237; 333968; 334389; 337612; 349670; for ILSVRC.
iid = randsample( iid', 1 ); 
db.demoBbox( 1, [ 3, 6, 1, 1 ], iid );
attNet.demoDet( iid, true );

%% CLEAN UP.
caffe.reset_all(  );



















