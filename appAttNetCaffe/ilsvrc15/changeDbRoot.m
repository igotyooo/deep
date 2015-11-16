clc; clear all; fclose all; close all;
addpath( genpath( '../../' ) ); init_ilsvrc15;

dstRoot.nvidia  = '/disk1/IMAGENET/ILSVRC2015/';
dstRoot.saturnw = '/data1/IMAGENET/ILSVRC2015/';
dstRoot.saturnx = '/data/IMAGENET/ILSVRC2015/';
dstRoot.saturny = '/data/IMAGENET/ILSVRC2015/';
dstRoot.saturnv = '/data1/IMAGENET/ILSVRC2015/';
srcRoot         = '/data/IMAGENET/ILSVRC2015/';
% srcRoot         = '/iron/db/ILSVRC2015/';
setting.db = path.db.ilsvrcclsloc2015te;
target = 'saturnv';

%% DO THE JOB.
db = Db( setting.db, path.dstDir );
db.genDb;

switch target,
    case 'nvidia',
        dstRoot = dstRoot.nvidia;
    case 'saturnw',
        dstRoot = dstRoot.saturnw;
    case 'saturnx'
        dstRoot = dstRoot.saturnx;
    case 'saturny'
        dstRoot = dstRoot.saturny;
    case 'saturnv'
        dstRoot = dstRoot.saturnv;
end;
if srcRoot( end ) == '/', srcRoot = srcRoot( 1 : end - 1 ); end;
if dstRoot( end ) == '/', dstRoot = dstRoot( 1 : end - 1 ); end;
srcRootLen = numel( srcRoot );
fprintf( 'Change path.\n' );
iid2impathSrc = db.iid2impath;
iid2impathDst = cell( db.getNumIm, 1 );
parfor iid = 1 : db.getNumIm;
    srcPath = iid2impathSrc{ iid };
    dstPath = fullfile( dstRoot, srcPath( srcRootLen + 1 : end ) );
    iid2impathDst{ iid } = dstPath;
end;
fprintf( 'Done.\n' );
fprintf( 'Save db.\n' );
fpath = fullfile( db.dstDir, strcat( 'DB_', upper( target ), '.mat' ) );
db_.iid2impath = iid2impathDst;
db_.cid2name = db.cid2name;
db_.cid2diids = db.cid2diids;
db_.iid2size = single( db.iid2size );
db_.iid2cids = db.iid2cids;
db_.iid2oids = db.iid2oids;
db_.iid2setid = db.iid2setid;
db_.oid2cid = db.oid2cid;
db_.oid2diff = db.oid2diff;
db_.oid2iid = db.oid2iid;
db_.oid2bbox = single( round( db.oid2bbox ) );
db_.oid2cont = db.oid2cont;
db = db_;
save( fpath, 'db' );
fprintf( 'Done.\n' );

